"""
실시간 차량 소리 분류기
학습된 PyTorch 모델을 사용하여 마이크 입력을 실시간으로 분류합니다.
"""

import os
import numpy as np
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from collections import deque
import time
import serial


# ========================= 설정 =========================
class Config:
    """
    모델 학습 시 사용한 설정과 동일하게 유지

    주의: 이 설정값들은 모델 학습 시 사용한 것과 정확히 일치해야 합니다.
          값을 변경하면 모델이 제대로 작동하지 않을 수 있습니다.
    """
    # ===== 오디오 처리 파라미터 =====
    SAMPLE_RATE = 22050          # 샘플링 레이트 (Hz)
                                 # 일반적으로 22050Hz 또는 44100Hz 사용

    AUDIO_DURATION = 4.0         # 분석할 오디오 길이 (초)
                                 # 4초 단위로 오디오를 수집하여 분류

    # ===== 멜 스펙트로그램 파라미터 =====
    N_MELS = 128                 # 멜 스펙트로그램의 주파수 빈 개수
                                 # 높을수록 주파수 해상도 증가

    N_FFT = 2048                 # FFT(고속 푸리에 변환) 윈도우 크기
                                 # 주파수 분해능 결정

    HOP_LENGTH = 512             # 홉 길이 (FFT 윈도우 이동 간격)
                                 # 시간 해상도 결정

    F_MIN = 20                   # 분석할 최소 주파수 (Hz)
                                 # 사람의 가청 주파수 범위: 20Hz~20kHz

    F_MAX = 8000                 # 분석할 최대 주파수 (Hz)
                                 # 대부분의 음향 특징은 8kHz 이하에 존재

    SPEC_HEIGHT = 224            # 스펙트로그램 이미지 높이 (픽셀)
    SPEC_WIDTH = 224             # 스펙트로그램 이미지 너비 (픽셀)
                                 # CNN 입력 크기 (224x224)

    # ===== 실시간 수집 설정 =====
    CHUNK = 1024                 # 한 번에 읽을 오디오 프레임 수
                                 # 작을수록 지연 시간 감소, 클수록 안정성 증가

    FORMAT = pyaudio.paFloat32   # 오디오 데이터 형식
                                 # paFloat32: 32비트 부동소수점

    CHANNELS = 1                 # 오디오 채널 수
                                 # 1: 모노, 2: 스테레오

    # ===== 클래스 이름 정의 =====
    # UrbanSound8K 데이터셋의 10개 클래스
    # 주의: 인덱스와 클래스명의 매핑은 학습 시와 동일해야 합니다.
    CLASS_NAMES = {
        0: "air_conditioner",      # 에어컨 소리
        1: "car_horn",             # 자동차 경적
        2: "children_playing",     # 어린이 놀이 소리
        3: "dog_bark",             # 개 짖는 소리
        4: "drilling",             # 드릴 소리
        5: "engine_idling",        # 엔진 공회전 소리
        6: "gun_shot",             # 총소리
        7: "jackhammer",           # 착암기 소리
        8: "siren",                # 사이렌 소리
        9: "street_music"          # 거리 음악
    }


# 전역 Config 인스턴스
config = Config()


# ========================= 모델 정의 =========================
class SimpleCNN(nn.Module):
    """
    학습 시 사용한 모델과 동일한 CNN 구조

    4개의 합성곱 블록과 3개의 완전연결 레이어로 구성된 분류 모델입니다.
    각 블록은 합성곱, 배치정규화, ReLU 활성화, 풀링, 드롭아웃으로 구성됩니다.

    주의: 이 구조는 학습된 모델과 정확히 일치해야 합니다.
    """

    def __init__(self, num_classes=10):
        """
        Args:
            num_classes (int): 분류할 클래스 개수 (기본값: 10)
        """
        super().__init__()

        # ===== 특징 추출 레이어 =====
        self.features = nn.Sequential(
            # Block 1: 첫 번째 합성곱 블록
            nn.Conv2d(3, 64, 3, padding=1),      # 3채널(RGB) -> 64채널
            nn.BatchNorm2d(64),                   # 배치 정규화 (학습 안정화)
            nn.ReLU(inplace=True),                # ReLU 활성화 함수
            nn.Conv2d(64, 64, 3, padding=1),     # 64채널 유지
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 2x2 최대 풀링 (크기 1/2)
            nn.Dropout2d(0.1),                    # 10% 드롭아웃 (과적합 방지)

            # Block 2: 두 번째 합성곱 블록
            nn.Conv2d(64, 128, 3, padding=1),    # 64채널 -> 128채널
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),   # 128채널 유지
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/2
            nn.Dropout2d(0.2),                    # 20% 드롭아웃

            # Block 3: 세 번째 합성곱 블록
            nn.Conv2d(128, 256, 3, padding=1),   # 128채널 -> 256채널
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),   # 256채널 유지
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/2
            nn.Dropout2d(0.2),

            # Block 4: 네 번째 합성곱 블록 (가장 깊은 특징)
            nn.Conv2d(256, 512, 3, padding=1),   # 256채널 -> 512채널
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),   # 512채널 유지
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/2
            nn.Dropout2d(0.3),                    # 30% 드롭아웃
        )

        # ===== 적응형 평균 풀링 =====
        # 입력 크기에 관계없이 출력을 7x7로 고정
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ===== 분류 레이어 =====
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),        # 25088 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                      # 50% 드롭아웃
            nn.Linear(1024, 512),                 # 1024 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)           # 512 -> 10 (클래스 수)
        )

    def forward(self, x):
        """
        순전파 함수

        Args:
            x (torch.Tensor): 입력 텐서 (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: 클래스별 로짓 값 (batch_size, num_classes)
        """
        x = self.features(x)         # 특징 추출
        x = self.avgpool(x)          # 적응형 풀링
        x = torch.flatten(x, 1)      # 1차원으로 펼침 (배치 차원 제외)
        x = self.classifier(x)       # 분류
        return x


# ========================= 실시간 분류기 =========================
class RealtimeAudioClassifier:
    """
    실시간 오디오 분류기 (배치 수집 방식)

    4초 단위로 오디오를 수집하고 분류합니다.
    타겟 라벨 감지 시 시리얼 통신으로 결과를 전송할 수 있습니다.
    """

    def __init__(self, model_path, target_labels=None, confidence_threshold=0.5,
                 serial_port=None, baud_rate=9600):
        """
        Args:
            model_path (str): 학습된 모델 파일 경로 (.pth 파일)
            target_labels (list): 감지할 타겟 라벨 리스트
                                 예시: ['car_horn', 'siren', 'engine_idling']
                                 None 또는 빈 리스트면 모든 클래스 감지
            confidence_threshold (float): 타겟 감지를 위한 최소 신뢰도 (0.0~1.0)
                                         예시: 0.5 = 50% 이상의 신뢰도 필요
            serial_port (str): 시리얼 포트 이름
                              Windows: 'COM3', 'COM4' 등
                              Linux: '/dev/ttyUSB0', '/dev/ttyACM0' 등
                              Mac: '/dev/cu.usbserial-1420' 등
                              None이면 시리얼 통신 비활성화
            baud_rate (int): 시리얼 통신 속도 (기본값: 9600)
                            일반적인 값: 9600, 19200, 38400, 57600, 115200
        """
        # ===== 디바이스 설정 =====
        # GPU 사용 가능 시 cuda, 아니면 cpu 사용
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ===== 타겟 라벨 설정 =====
        # 감지할 특정 소리 클래스 지정
        self.target_labels = target_labels or []

        # ===== 신뢰도 임계값 설정 =====
        # 이 값 이상의 신뢰도를 가진 예측만 타겟으로 인식
        self.confidence_threshold = confidence_threshold

        # ===== 모델 로드 =====
        print(f"Loading model: {model_path}")

        # 체크포인트 파일 로드
        # map_location: 저장된 디바이스와 다른 디바이스에서도 로드 가능
        checkpoint = torch.load(model_path, map_location=self.device)

        # 모델 인스턴스 생성
        self.model = SimpleCNN(num_classes=10)

        # 학습된 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 모델을 지정된 디바이스로 이동
        self.model.to(self.device)

        # 평가 모드로 설정 (드롭아웃, 배치정규화 등이 추론 모드로 동작)
        self.model.eval()

        print(f"Model loaded successfully")
        print(f"   Device: {self.device}")
        if 'best_acc' in checkpoint:
            print(f"   Training accuracy: {checkpoint['best_acc']:.2f}%")

        # ===== 멜 스펙트로그램 변환기 초기화 =====
        # 오디오 파형을 멜 스펙트로그램으로 변환
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )

        # ===== PyAudio 초기화 =====
        self.p = pyaudio.PyAudio()
        self.stream = None

        # ===== 시리얼 포트 설정 =====
        self.serial_conn = None           # 시리얼 연결 객체
        self.serial_enabled = False       # 시리얼 연결 상태 플래그

        if serial_port:
            # 시리얼 포트가 지정된 경우 연결 시도
            try:
                self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)
                time.sleep(2)  # 연결 안정화를 위한 대기
                self.serial_enabled = True
                print(f"Serial port connected: {serial_port} @ {baud_rate} bps")
            except Exception as e:
                print(f"Serial port connection failed: {e}")
                self.serial_conn = None
                self.serial_enabled = False
        else:
            # 시리얼 포트가 지정되지 않은 경우
            print("Serial port: NONE (not configured)")

        # ===== 통계 변수 초기화 =====
        # 각 타겟 라벨의 감지 횟수 추적
        self.detection_count = {label: 0 for label in self.target_labels}
        # 전체 예측 횟수
        self.total_predictions = 0

    def send_to_serial(self, class_id, class_name, confidence):
        """
        시리얼 포트로 분류 결과 전송

        전송 형식: "CLASS_ID,CLASS_NAME,CONFIDENCE\n"
        예시: "1,car_horn,85.32\n"

        이 형식은 Arduino나 다른 마이크로컨트롤러에서 쉽게 파싱할 수 있습니다.

        Arduino 수신 예시:
```
        String data = Serial.readStringUntil('\n');
        int commaIndex1 = data.indexOf(',');
        int commaIndex2 = data.indexOf(',', commaIndex1 + 1);

        int classId = data.substring(0, commaIndex1).toInt();
        String className = data.substring(commaIndex1 + 1, commaIndex2);
        float confidence = data.substring(commaIndex2 + 1).toFloat();
```

        Args:
            class_id (int): 클래스 ID (0~9)
            class_name (str): 클래스 이름
            confidence (float): 신뢰도 (0.0~100.0)

        Returns:
            str: 전송 상태 메시지
                 "NONE": 시리얼 비활성화
                 "SENT: ...": 전송 성공
                 "ERROR: ...": 전송 실패
                 "DISCONNECTED": 포트 연결 끊김
        """
        # 시리얼이 비활성화된 경우
        if not self.serial_enabled:
            return "NONE"

        # 시리얼 포트가 열려있는지 확인
        if self.serial_conn and self.serial_conn.is_open:
            try:
                # 메시지 형식 생성 (CSV 형식)
                message = f"{class_id},{class_name},{confidence:.2f}\n"

                # 시리얼로 전송 (문자열을 바이트로 인코딩)
                self.serial_conn.write(message.encode())

                return f"SENT: {message.strip()}"
            except Exception as e:
                return f"ERROR: {e}"
        else:
            return "DISCONNECTED"

    def preprocess_audio(self, audio_data):
        """
        오디오 데이터를 모델 입력 형식으로 전처리

        처리 과정:
        1. NumPy 배열 -> PyTorch 텐서 변환
        2. 멜 스펙트로그램 생성 (시간-주파수 변환)
        3. dB 스케일로 변환 (사람의 청각 특성 반영)
        4. 정규화 (평균 0, 표준편차 1)
        5. 크기 조정 (224x224로 리사이즈)
        6. 3채널로 복제 (CNN 입력 형식에 맞춤)

        Args:
            audio_data (numpy.ndarray): 원본 오디오 데이터 (1차원 배열)

        Returns:
            torch.Tensor: 전처리된 텐서 (1, 3, 224, 224)
        """
        # ===== NumPy -> PyTorch 텐서 변환 =====
        # unsqueeze(0): 배치 차원 추가 (1, length)
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

        # ===== 멜 스펙트로그램 생성 =====
        # 시간 도메인 -> 시간-주파수 도메인
        mel_spec = self.mel_transform(waveform)

        # dB 스케일로 변환 (로그 스케일)
        # 사람의 청각은 소리 강도를 로그 스케일로 인지
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # ===== 정규화 =====
        # 평균 0, 표준편차 1로 정규화하여 학습 시와 동일한 분포 유지
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)  # 1e-8: 0 나누기 방지

        # ===== 크기 조정 =====
        # 모델 입력 크기 224x224로 리사이즈
        if mel_spec_db.shape[-1] != config.SPEC_WIDTH or mel_spec_db.shape[-2] != config.SPEC_HEIGHT:
            mel_spec_db = F.interpolate(
                mel_spec_db.unsqueeze(0),           # 배치 차원 추가
                size=(config.SPEC_HEIGHT, config.SPEC_WIDTH),
                mode='bilinear',                     # 이중선형 보간
                align_corners=False
            ).squeeze(0)                             # 배치 차원 제거

        # ===== 3채널로 복제 =====
        # CNN 모델이 RGB 이미지 형식을 기대하므로
        # 단일 채널 스펙트로그램을 3번 복제
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        # 배치 차원 추가 후 반환 (1, 3, 224, 224)
        return mel_spec_db.unsqueeze(0)

    def predict(self, audio_data):
        """
        오디오 데이터 분류

        Args:
            audio_data (numpy.ndarray): 원본 오디오 데이터

        Returns:
            tuple: (predicted_class, confidence, probabilities)
                - predicted_class (int): 예측된 클래스 ID (0~9)
                - confidence (float): 예측 신뢰도 (0.0~1.0)
                - probabilities (torch.Tensor): 모든 클래스의 확률 분포 (10,)
        """
        # 전처리
        input_tensor = self.preprocess_audio(audio_data).to(self.device)

        # ===== 모델 추론 =====
        with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약, 속도 향상)
            # 순전파
            outputs = self.model(input_tensor)

            # Softmax로 확률로 변환
            # dim=1: 클래스 차원에 대해 Softmax 적용
            probabilities = F.softmax(outputs, dim=1)[0]

            # 가장 높은 확률의 클래스 선택
            predicted_class = torch.argmax(probabilities).item()

            # 해당 클래스의 신뢰도
            confidence = probabilities[predicted_class].item()

        return predicted_class, confidence, probabilities

    def start_realtime_detection(self):
        """
        실시간 감지 시작 (배치 수집 방식)

        4초 단위로 오디오를 수집하고 분류합니다.
        - 4초 수집 -> 예측 -> 결과 출력 -> 다시 4초 수집 (반복)
        - Ctrl+C를 누르면 종료됩니다.
        """
        # ===== 시작 정보 출력 =====
        print(f"\n{'=' * 70}")
        print(f"Real-time Audio Classification Started")
        print(f"{'=' * 70}")
        print(f"Sample rate: {config.SAMPLE_RATE} Hz")
        print(f"Audio duration: {config.AUDIO_DURATION} sec")
        print(f"Target labels: {self.target_labels if self.target_labels else 'All classes'}")
        print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        print(f"Serial port: {'ENABLED' if self.serial_enabled else 'DISABLED'}")
        print(f"{'=' * 70}\n")

        # ===== 오디오 스트림 시작 =====
        self.stream = self.p.open(
            format=config.FORMAT,              # 오디오 형식 (Float32)
            channels=config.CHANNELS,          # 채널 수 (모노)
            rate=config.SAMPLE_RATE,           # 샘플링 레이트 (22050Hz)
            input=True,                        # 입력 스트림
            frames_per_buffer=config.CHUNK     # 버퍼 크기 (1024)
        )

        print("Audio collection started... (Press Ctrl+C to stop)\n")

        try:
            while True:
                # ===== 오디오 데이터 수집 (4초) =====
                data_buffer = []

                # 4초 분량의 청크 수 계산
                # 예: 22050 샘플/초 * 4초 / 1024 샘플/청크 ≈ 86 청크
                chunks_needed = int(config.SAMPLE_RATE / config.CHUNK * config.AUDIO_DURATION)

                # 수집 상태 표시
                print('Collecting audio...', end=' ', flush=True)

                # 청크 단위로 오디오 읽기
                for _ in range(chunks_needed):
                    # exception_on_overflow=False: 버퍼 오버플로우 시 경고만 출력
                    data = self.stream.read(config.CHUNK, exception_on_overflow=False)
                    data_buffer.append(data)

                # ===== 데이터 변환 =====
                # 바이트 데이터 결합
                audio_data = b''.join(data_buffer)

                # NumPy 배열로 변환 (float32)
                audio_data = np.frombuffer(audio_data, dtype=np.float32)

                # ===== 분류 수행 =====
                predicted_class, confidence, probabilities = self.predict(audio_data)
                class_name = config.CLASS_NAMES[predicted_class]

                # ===== 통계 업데이트 =====
                self.total_predictions += 1

                # ===== 결과 출력 =====
                # 분류 결과와 신뢰도 출력
                result_str = f'{class_name:20s} {confidence * 100:6.2f}%'
                print(result_str)

                # ===== 타겟 라벨 감지 확인 =====
                # 타겟 라벨이고 신뢰도가 임계값 이상인 경우
                is_target_detected = (class_name in self.target_labels and
                                     confidence >= self.confidence_threshold)

                if is_target_detected:
                    # ===== 타겟 감지됨 =====
                    # 감지 횟수 증가
                    self.detection_count[class_name] += 1

                    # 시리얼 전송
                    serial_status = self.send_to_serial(predicted_class, class_name, confidence * 100)

                    # 강조 메시지 출력
                    print(f'[DETECTED: {class_name.upper()}] [Confidence: {confidence * 100:.2f}%]')
                    print(f'[Serial: {serial_status}]')
                else:
                    # ===== 타겟 아님 =====
                    if self.serial_enabled:
                        # 시리얼이 활성화되어 있지만 타겟이 아니거나 신뢰도 부족
                        print(f'[Serial: NONE (Not target or low confidence)]')
                    else:
                        # 시리얼이 비활성화됨
                        print(f'[Serial: NONE]')

                # ===== 상위 3개 클래스 표시 =====
                # 신뢰도가 높은 순서대로 상위 3개 클래스 출력
                top3_indices = torch.topk(probabilities, 3).indices
                print('   Top 3:', end=' ')
                for idx in top3_indices:
                    idx = idx.item()
                    print(f'{config.CLASS_NAMES[idx]}: {probabilities[idx] * 100:.1f}%', end=' | ')
                print('\n')

                # 짧은 대기 (다음 수집 전 안정화)
                time.sleep(0.1)

        except KeyboardInterrupt:
            # Ctrl+C를 누르면 종료
            print("\n\nShutting down...")
            self.stop()
            self.print_statistics()

    def stop(self):
        """
        스트림 및 연결 종료

        오디오 스트림과 시리얼 포트를 안전하게 종료합니다.
        """
        # 오디오 스트림 종료
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

        # 시리얼 포트 종료
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial port closed")

        print("Audio stream closed")

    def print_statistics(self):
        """
        감지 통계 출력

        프로그램 종료 시 타겟 라벨의 감지 횟수와 비율을 출력합니다.
        """
        print(f"\n{'=' * 70}")
        print(f"Detection Statistics")
        print(f"{'=' * 70}")
        print(f"Total predictions: {self.total_predictions}")

        if self.target_labels:
            print(f"\nTarget label detection count:")
            for label in self.target_labels:
                count = self.detection_count[label]
                # 전체 예측 중 해당 라벨의 감지 비율 계산
                percentage = (count / self.total_predictions * 100) if self.total_predictions > 0 else 0
                print(f"   {label:20s}: {count:4d} times ({percentage:5.2f}%)")

        print(f"{'=' * 70}\n")


# ========================= 메인 실행 =========================
def main():
    """
    메인 함수

    실시간 오디오 분류기를 설정하고 실행합니다.

    ===== 사용자 설정 가이드 =====

    1. 모델 경로 변경:
       MODEL_PATH를 실제 모델 파일 경로로 변경하세요.

    2. 감지할 소리 변경 (TARGET_LABELS):
       감지하려는 클래스를 리스트로 지정하세요.

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
       TARGET_LABELS = ['car_horn', 'siren']              # 경적과 사이렌만
       TARGET_LABELS = ['dog_bark', 'children_playing']   # 개와 어린이 소리만
       TARGET_LABELS = []                                  # 모든 소리 감지

    3. 신뢰도 임계값 변경 (CONFIDENCE_THRESHOLD):
       타겟 감지를 위한 최소 신뢰도를 설정하세요.

       예시:
       CONFIDENCE_THRESHOLD = 0.5    # 50% 이상 (기본값, 일반적)
       CONFIDENCE_THRESHOLD = 0.7    # 70% 이상 (정확도 중시)
       CONFIDENCE_THRESHOLD = 0.3    # 30% 이상 (감지율 중시)

    4. 시리얼 포트 설정 (SERIAL_PORT):
       실제 연결된 포트명으로 변경하세요.

       Windows 예시:
       SERIAL_PORT = 'COM3'         # 장치 관리자에서 확인
       SERIAL_PORT = 'COM4'

       Linux 예시:
       SERIAL_PORT = '/dev/ttyUSB0'  # dmesg | grep tty 로 확인
       SERIAL_PORT = '/dev/ttyACM0'

       Mac 예시:
       SERIAL_PORT = '/dev/cu.usbserial-1420'
       SERIAL_PORT = '/dev/cu.usbmodem14201'

       시리얼 통신 사용 안 함:
       SERIAL_PORT = None

    5. 시리얼 통신 속도 변경 (BAUD_RATE):
       연결된 장치와 동일한 속도로 설정하세요.

       일반적인 값:
       BAUD_RATE = 9600      # 가장 일반적 (Arduino 기본값)
       BAUD_RATE = 19200
       BAUD_RATE = 38400
       BAUD_RATE = 57600
       BAUD_RATE = 115200    # 고속 통신
    """
    # ===== 모델 경로 설정 =====
    # 학습된 모델 파일 경로를 지정하세요.
    MODEL_PATH = './saved_models/vehicle_audio_simple_integration_best.pth'

    # ===== 타겟 라벨 설정 =====
    # 감지하고 싶은 소리를 리스트로 지정하세요.
    # 차량 관련 소리 3가지를 감지합니다.
    TARGET_LABELS = ['car_horn', 'engine_idling', 'siren']

    # 다른 예시:
    # TARGET_LABELS = ['dog_bark']                       # 개 짖는 소리만
    # TARGET_LABELS = ['siren', 'gun_shot']              # 사이렌과 총소리
    # TARGET_LABELS = []                                  # 모든 클래스 감지

    # ===== 신뢰도 임계값 설정 =====
    # 타겟으로 인식하기 위한 최소 신뢰도 (0.0 ~ 1.0)
    CONFIDENCE_THRESHOLD = 0.5

    # 조정 예시:
    # CONFIDENCE_THRESHOLD = 0.7  # 높은 정확도 필요 시
    # CONFIDENCE_THRESHOLD = 0.3  # 높은 감지율 필요 시

    # ===== 시리얼 포트 설정 =====
    # 실제 포트명으로 변경하세요.
    # 현재는 None으로 시리얼 통신 비활성화됨.
    SERIAL_PORT = None  # 시리얼 통신 사용 안 함

    # 실제 사용 예시 (주석 제거 후 사용):
    # SERIAL_PORT = 'COM3'                    # Windows
    # SERIAL_PORT = '/dev/ttyUSB0'            # Linux
    # SERIAL_PORT = '/dev/cu.usbserial-1420'  # Mac

    # ===== 시리얼 통신 속도 설정 =====
    BAUD_RATE = 9600

    # 다른 속도 예시:
    # BAUD_RATE = 115200  # 고속 통신 필요 시

    # ===== 모델 파일 존재 확인 =====
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print(f"Please set the correct path.")
        return

    # ===== 분류기 생성 =====
    classifier = RealtimeAudioClassifier(
        model_path=MODEL_PATH,
        target_labels=TARGET_LABELS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        serial_port=SERIAL_PORT,
        baud_rate=BAUD_RATE
    )

    # ===== 실시간 감지 시작 =====
    classifier.start_realtime_detection()


if __name__ == '__main__':
    main()