"""
실시간 오디오 분류기 - PyTorch 버전
학습된 차량 소리 분류 모델을 사용하여 실시간으로 오디오를 분류합니다.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pyaudio
import time
import serial
import serial.tools.list_ports


# ========================= SimpleCNN 모델 정의 =========================
class SimpleCNN(nn.Module):
    """
    학습에 사용한 것과 동일한 CNN 모델 구조

    주의: 이 구조는 학습 시 사용한 모델과 정확히 일치해야 합니다.
          레이어 구조나 파라미터를 변경하면 학습된 가중치를 로드할 수 없습니다.
    """

    def __init__(self, num_classes=10):
        """
        Args:
            num_classes (int): 분류할 클래스 개수 (기본값: 10)
                              UrbanSound8K 데이터셋 기준 10개 클래스
        """
        super().__init__()

        # 특징 추출 레이어 (Convolutional layers)
        self.features = nn.Sequential(
            # Block 1: 첫 번째 합성곱 블록
            nn.Conv2d(3, 64, 3, padding=1),      # 입력 3채널 -> 64채널
            nn.BatchNorm2d(64),                   # 배치 정규화
            nn.ReLU(inplace=True),                # 활성화 함수
            nn.Conv2d(64, 64, 3, padding=1),     # 64채널 -> 64채널
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/2로 축소
            nn.Dropout2d(0.1),                    # 과적합 방지 (10% 드롭아웃)

            # Block 2: 두 번째 합성곱 블록
            nn.Conv2d(64, 128, 3, padding=1),    # 64채널 -> 128채널
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),   # 128채널 -> 128채널
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/2로 축소
            nn.Dropout2d(0.2),                    # 과적합 방지 (20% 드롭아웃)

            # Block 3: 세 번째 합성곱 블록
            nn.Conv2d(128, 256, 3, padding=1),   # 128채널 -> 256채널
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),   # 256채널 -> 256채널
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/2로 축소
            nn.Dropout2d(0.2),

            # Block 4: 네 번째 합성곱 블록 (가장 깊은 레이어)
            nn.Conv2d(256, 512, 3, padding=1),   # 256채널 -> 512채널
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),   # 512채널 -> 512채널
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/2로 축소
            nn.Dropout2d(0.3),                    # 과적합 방지 (30% 드롭아웃)
        )

        # 적응형 평균 풀링 (특징 맵 크기를 7x7로 고정)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 분류 레이어 (Fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),        # 512*7*7 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                      # 과적합 방지 (50% 드롭아웃)
            nn.Linear(1024, 512),                 # 1024 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)           # 512 -> 10 (최종 클래스 수)
        )

    def forward(self, x):
        """
        순전파 함수

        Args:
            x (torch.Tensor): 입력 텐서 (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: 클래스별 점수 (batch_size, num_classes)
        """
        x = self.features(x)         # 특징 추출
        x = self.avgpool(x)          # 적응형 풀링
        x = torch.flatten(x, 1)      # 1차원으로 펼침
        x = self.classifier(x)       # 분류
        return x


# ========================= 실시간 분류기 =========================
class RealtimeAudioClassifier:
    """
    실시간 오디오 분류기

    PyAudio를 사용하여 마이크로부터 오디오를 수집하고,
    학습된 모델로 실시간 분류를 수행합니다.
    선택적으로 시리얼 통신을 통해 결과를 외부 장치로 전송할 수 있습니다.
    """

    def __init__(self, model_path, target_labels=None, serial_port=None, baud_rate=9600):
        """
        Args:
            model_path (str): 학습된 모델 파일 경로 (.pth 파일)
            target_labels (list): 감지할 타겟 라벨 리스트
                                 예시: ['car_horn', 'siren', 'engine_idling']
                                 None이면 기본값 사용
            serial_port (str): 시리얼 포트 이름
                              Windows: 'COM3', 'COM4' 등
                              Linux: '/dev/ttyUSB0', '/dev/ttyACM0' 등
                              Mac: '/dev/cu.usbserial-1420' 등
                              None이면 시리얼 통신 사용 안 함
            baud_rate (int): 시리얼 통신 속도 (기본값: 9600)
                            일반적으로 9600, 19200, 38400, 57600, 115200 사용
        """
        # ===== 오디오 처리 설정 =====
        # 주의: 이 값들은 모델 학습 시 사용한 설정과 동일해야 합니다.
        self.SAMPLE_RATE = 22050              # 샘플링 레이트 (Hz)
        self.AUDIO_DURATION = 4.0             # 오디오 길이 (초)
        self.N_MELS = 128                     # 멜 스펙트로그램 주파수 빈 개수
        self.N_FFT = 2048                     # FFT 윈도우 크기
        self.HOP_LENGTH = 512                 # 홉 길이 (FFT 윈도우 이동 간격)
        self.F_MIN = 20                       # 최소 주파수 (Hz)
        self.F_MAX = 8000                     # 최대 주파수 (Hz)
        self.SPEC_HEIGHT = 224                # 스펙트로그램 높이 (픽셀)
        self.SPEC_WIDTH = 224                 # 스펙트로그램 너비 (픽셀)

        # ===== 클래스 이름 정의 =====
        # UrbanSound8K 데이터셋의 10개 클래스
        # 주의: 인덱스 순서는 학습 시 사용한 것과 동일해야 합니다.
        self.CLASS_NAMES = {
            0: "air_conditioner",      # 에어컨 소리
            1: "car_horn",             # 자동차 경적 소리
            2: "children_playing",     # 어린이 놀이 소리
            3: "dog_bark",             # 개 짖는 소리
            4: "drilling",             # 드릴 소리
            5: "engine_idling",        # 엔진 공회전 소리
            6: "gun_shot",             # 총소리
            7: "jackhammer",           # 착암기 소리
            8: "siren",                # 사이렌 소리
            9: "street_music"          # 거리 음악 소리
        }

        # ===== 타겟 라벨 설정 =====
        # 감지하고자 하는 특정 소리 설정
        # 사용자 정의: 원하는 클래스를 리스트로 지정
        # 예시 1: ['car_horn', 'siren'] - 경적과 사이렌만 감지
        # 예시 2: ['dog_bark', 'gun_shot'] - 개 짖는 소리와 총소리만 감지
        # 예시 3: None - 모든 클래스 감지
        if target_labels is None:
            self.target_labels = ['car_horn', 'engine_idling', 'siren']
        else:
            self.target_labels = target_labels

        # ===== 시리얼 통신 초기화 =====
        self.serial_conn = None
        if serial_port:
            # 시리얼 포트가 지정된 경우
            try:
                # 시리얼 포트 연결 시도
                # timeout=1: 1초 대기 후 타임아웃
                self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)
                print(f'[시리얼 포트 연결: {serial_port} @ {baud_rate}bps]')
            except Exception as e:
                print(f'[시리얼 포트 연결 실패: {e}]')
                self.serial_conn = None
        else:
            # 시리얼 포트가 지정되지 않은 경우
            # 사용 가능한 포트 목록 출력
            print('\n[사용 가능한 시리얼 포트]')
            ports = serial.tools.list_ports.comports()
            if ports:
                for port in ports:
                    print(f'  - {port.device}: {port.description}')
            else:
                print('  - 사용 가능한 포트가 없습니다.')

        # ===== PyTorch 디바이스 설정 =====
        # GPU가 사용 가능하면 cuda, 아니면 cpu 사용
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n[Device: {self.device}]')

        # ===== 모델 로드 =====
        print(f'[모델 로딩: {model_path}]')

        # 학습된 모델 체크포인트 로드
        # map_location: 모델을 현재 디바이스에 맞게 로드
        checkpoint = torch.load(model_path, map_location=self.device)

        # 모델 인스턴스 생성
        self.model = SimpleCNN(num_classes=10)

        # 학습된 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 모델을 지정된 디바이스로 이동
        self.model.to(self.device)

        # 평가 모드 설정 (드롭아웃, 배치정규화 등이 추론 모드로 동작)
        self.model.eval()

        print(f'[모델 로드 완료 - 정확도: {checkpoint["best_acc"]:.2f}%]')

        # ===== 멜 스펙트로그램 변환기 초기화 =====
        # 오디오 파형을 멜 스펙트로그램으로 변환하는 객체
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            f_min=self.F_MIN,
            f_max=self.F_MAX
        )

        # ===== PyAudio 설정 =====
        # 실시간 오디오 수집을 위한 설정
        self.CHUNK = 1024                      # 한 번에 읽을 오디오 프레임 수
        self.FORMAT = pyaudio.paFloat32        # 오디오 데이터 형식 (32비트 float)
        self.CHANNELS = 1                      # 모노 채널 (스테레오는 2)

        # 초기화 정보 출력
        print(f'\n[타겟 라벨: {self.target_labels}]')
        print(f'[오디오 설정: {self.SAMPLE_RATE}Hz, {self.AUDIO_DURATION}초]')
        print(f'[클래스: {len(self.CLASS_NAMES)}개]')

    def send_to_serial(self, class_id, class_name, confidence):
        """
        시리얼 포트로 분류 결과 전송

        전송 형식: "CLASS_ID,CLASS_NAME,CONFIDENCE\n"
        예시: "1,car_horn,85.32\n"

        Args:
            class_id (int): 클래스 ID (0~9)
            class_name (str): 클래스 이름
            confidence (float): 신뢰도 (0~100)
        """
        if self.serial_conn and self.serial_conn.is_open:
            try:
                # 메시지 형식 생성
                # Arduino나 다른 마이크로컨트롤러에서 파싱하기 쉬운 형식
                message = f"{class_id},{class_name},{confidence:.2f}\n"

                # 시리얼 포트로 전송 (문자열을 바이트로 인코딩)
                self.serial_conn.write(message.encode())

                print(f'[시리얼 전송: {message.strip()}]')
            except Exception as e:
                print(f'[시리얼 전송 실패: {e}]')

    def preprocess_audio(self, audio_data):
        """
        오디오 데이터를 모델 입력 형식으로 전처리

        처리 순서:
        1. NumPy 배열을 PyTorch 텐서로 변환
        2. 오디오 길이를 4초로 맞춤 (자르거나 패딩)
        3. 멜 스펙트로그램 생성
        4. dB 스케일로 변환
        5. 정규화
        6. 크기를 224x224로 조정
        7. 3채널로 복제

        Args:
            audio_data (numpy.ndarray): 원본 오디오 데이터 (1D 배열)

        Returns:
            torch.Tensor: 전처리된 텐서 (1, 3, 224, 224)
        """
        # NumPy 배열을 PyTorch 텐서로 변환 및 배치 차원 추가
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

        # ===== 오디오 길이 조정 =====
        # 모델은 항상 4초 길이의 오디오를 기대함
        target_length = int(self.SAMPLE_RATE * self.AUDIO_DURATION)

        if waveform.shape[1] > target_length:
            # 오디오가 4초보다 길면 중앙 부분 추출
            start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        else:
            # 오디오가 4초보다 짧으면 0으로 패딩
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        # ===== 멜 스펙트로그램 생성 =====
        # 시간-주파수 도메인으로 변환
        mel_spec = self.mel_transform(waveform)

        # dB 스케일로 변환 (사람의 청각 특성 반영)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # ===== 정규화 =====
        # 평균 0, 표준편차 1로 정규화
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)  # 1e-8: 0으로 나누기 방지

        # ===== 크기 조정 =====
        # 모델 입력 크기인 224x224로 조정
        mel_spec_db = F.interpolate(
            mel_spec_db.unsqueeze(0),
            size=(self.SPEC_HEIGHT, self.SPEC_WIDTH),
            mode='bilinear',              # 이중선형 보간
            align_corners=False
        ).squeeze(0)

        # ===== 3채널로 복제 =====
        # CNN 모델이 RGB 이미지 형식(3채널)을 기대하므로
        # 단일 채널을 3번 복제
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        # 배치 차원 추가 (모델 입력 형식에 맞춤)
        return mel_spec_db.unsqueeze(0)

    def predict(self, audio_data):
        """
        오디오 데이터 예측

        Args:
            audio_data (numpy.ndarray): 원본 오디오 데이터

        Returns:
            tuple: (predicted_class, confidence, probabilities)
                - predicted_class (int): 예측된 클래스 ID
                - confidence (float): 예측 신뢰도 (0~1)
                - probabilities (torch.Tensor): 모든 클래스의 확률 분포
        """
        # 오디오 데이터 전처리
        input_tensor = self.preprocess_audio(audio_data).to(self.device)

        # ===== 모델 추론 =====
        with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
            # 모델 순전파
            outputs = self.model(input_tensor)

            # Softmax를 적용하여 확률로 변환
            probabilities = F.softmax(outputs, dim=1)[0]

            # 가장 높은 확률을 가진 클래스 선택
            predicted_class = torch.argmax(probabilities).item()

            # 해당 클래스의 신뢰도
            confidence = probabilities[predicted_class].item()

        return predicted_class, confidence, probabilities

    def start_realtime_classification(self):
        """
        실시간 분류 시작

        마이크로부터 오디오를 수집하고 4초마다 분류를 수행합니다.
        Ctrl+C를 누르면 종료됩니다.
        """
        # PyAudio 인스턴스 생성
        p = pyaudio.PyAudio()

        # ===== 사용 가능한 입력 디바이스 목록 출력 =====
        print(f'\n[사용 가능한 입력 디바이스]')
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            # 입력 채널이 있는 디바이스만 출력
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")

        # ===== 오디오 스트림 열기 =====
        try:
            stream = p.open(
                format=self.FORMAT,              # 오디오 형식
                channels=self.CHANNELS,          # 채널 수 (모노)
                rate=self.SAMPLE_RATE,           # 샘플링 레이트
                input=True,                      # 입력 스트림
                frames_per_buffer=self.CHUNK     # 버퍼 크기
            )
        except Exception as e:
            print(f'[오디오 스트림 열기 실패: {e}]')
            p.terminate()
            return

        # 시작 메시지 출력
        print(f'\n{"=" * 70}')
        print(f'[실시간 오디오 분류 시작]')
        print(f'{"=" * 70}')
        print(f'종료하려면 Ctrl+C를 누르세요.\n')

        try:
            while True:
                # ===== 오디오 데이터 수집 =====
                data_buffer = []

                # 4초 분량의 샘플 수 계산
                samples_needed = int(self.SAMPLE_RATE * self.AUDIO_DURATION)

                # 필요한 청크 수 계산
                chunks_needed = int(samples_needed / self.CHUNK)

                # 수집 상태 표시
                print('[오디오 수집 중...]', end='', flush=True)

                # 청크 단위로 오디오 읽기
                for i in range(chunks_needed):
                    try:
                        # 오디오 청크 읽기
                        # exception_on_overflow=False: 버퍼 오버플로우 시 예외 발생 안 함
                        data = stream.read(self.CHUNK, exception_on_overflow=False)
                        data_buffer.append(data)
                    except Exception as e:
                        print(f'\n[오디오 읽기 오류: {e}]')
                        continue

                # ===== 데이터 결합 및 변환 =====
                # 바이트 데이터 결합
                audio_data = b''.join(data_buffer)

                # NumPy 배열로 변환 (float32)
                audio_data = np.frombuffer(audio_data, dtype=np.float32)

                # ===== 예측 수행 =====
                predicted_class, confidence, probabilities = self.predict(audio_data)
                class_name = self.CLASS_NAMES[predicted_class]

                # ===== 결과 출력 =====
                print(f'\r{"=" * 70}')
                print(f'[분류 결과] {class_name} ({confidence * 100:.2f}%)')

                # 타겟 라벨 감지 시 강조 메시지
                if class_name in self.target_labels:
                    print(f'*** DETECTED: {class_name.upper()} ***')

                # ===== 시리얼 전송 =====
                # 타겟 라벨 여부와 관계없이 모든 결과 전송
                self.send_to_serial(predicted_class, class_name, confidence * 100)

                print(f'{"=" * 70}\n')

                # 짧은 대기 (다음 수집 전)
                time.sleep(0.1)

        except KeyboardInterrupt:
            # Ctrl+C를 누르면 종료
            print(f'\n\n[사용자에 의해 중지됨]')

        except Exception as e:
            # 기타 오류 발생 시
            print(f'\n\n[오류 발생: {e}]')

        finally:
            # ===== 정리 =====
            print(f'[정리 중...]')

            # 오디오 스트림 종료
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 시리얼 포트 종료
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                print(f'[시리얼 포트 종료]')

            print(f'[종료 완료]')


# ========================= 메인 함수 =========================
def main():
    """
    메인 함수

    실시간 오디오 분류기를 설정하고 시작합니다.

    ===== 사용자 설정 가이드 =====

    1. 모델 경로 변경:
       model_path 변수를 사용하려는 모델 파일 경로로 변경하세요.

    2. 감지할 소리 변경:
       target_labels 리스트를 수정하세요.
       사용 가능한 클래스:
       - 'air_conditioner': 에어컨 소리
       - 'car_horn': 자동차 경적
       - 'children_playing': 어린이 놀이 소리
       - 'dog_bark': 개 짖는 소리
       - 'drilling': 드릴 소리
       - 'engine_idling': 엔진 공회전
       - 'gun_shot': 총소리
       - 'jackhammer': 착암기 소리
       - 'siren': 사이렌 소리
       - 'street_music': 거리 음악

       예시:
       target_labels = ['car_horn', 'siren']           # 경적과 사이렌만
       target_labels = ['dog_bark']                    # 개 짖는 소리만
       target_labels = None                            # 모든 소리 감지

    3. 시리얼 포트 설정:
       serial_port 변수를 실제 포트명으로 변경하세요.

       Windows 예시:
       serial_port = 'COM3'         # 장치 관리자에서 확인 가능
       serial_port = 'COM4'

       Linux 예시:
       serial_port = '/dev/ttyUSB0'  # dmesg | grep tty 로 확인 가능
       serial_port = '/dev/ttyACM0'

       Mac 예시:
       serial_port = '/dev/cu.usbserial-1420'
       serial_port = '/dev/cu.usbmodem14201'

       시리얼 통신 사용 안 함:
       serial_port = None

    4. 시리얼 통신 속도 변경:
       baud_rate 변수를 변경하세요.
       일반적인 값: 9600, 19200, 38400, 57600, 115200
       Arduino의 경우 9600이 기본값입니다.
    """
    # ===== 모델 경로 설정 =====
    # 학습된 모델 파일의 경로를 지정하세요.
    model_path = './saved_models/vehicle_audio_balanced_with_sonyc_best.pth'

    # ===== 모델 파일 존재 확인 =====
    if not os.path.exists(model_path):
        print(f'[오류] 모델 파일을 찾을 수 없습니다: {model_path}')
        print(f'모델을 먼저 학습시켜주세요.')
        return

    # ===== 타겟 라벨 설정 =====
    # 감지하고 싶은 소리를 리스트로 지정하세요.
    # 여기서는 차량 관련 소리 3가지를 감지합니다.
    target_labels = ['car_horn', 'engine_idling', 'siren']

    # 다른 예시:
    # target_labels = ['dog_bark', 'children_playing']  # 개와 어린이 소리 감지
    # target_labels = ['siren']                         # 사이렌만 감지
    # target_labels = None                              # 모든 클래스 감지

    # ===== 시리얼 포트 설정 =====
    # 실제 연결된 포트명으로 변경하세요.
    # 시리얼 통신을 사용하지 않으려면 None으로 설정하세요.
    serial_port = None  # 현재는 시리얼 통신 비활성화

    # 실제 사용 예시 (주석 제거 후 사용):
    # serial_port = 'COM3'                    # Windows
    # serial_port = '/dev/ttyUSB0'            # Linux
    # serial_port = '/dev/cu.usbserial-1420'  # Mac

    # ===== 시리얼 통신 속도 설정 =====
    # 연결된 장치와 동일한 속도로 설정해야 합니다.
    baud_rate = 9600

    # 다른 속도 예시:
    # baud_rate = 115200  # 고속 통신

    # ===== 분류기 생성 및 시작 =====
    classifier = RealtimeAudioClassifier(
        model_path=model_path,
        target_labels=target_labels,
        serial_port=serial_port,
        baud_rate=baud_rate
    )

    # 실시간 분류 시작
    classifier.start_realtime_classification()


if __name__ == '__main__':
    main()