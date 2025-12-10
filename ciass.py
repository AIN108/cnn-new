"""
실시간 차량 소리 분류기 - 슬라이딩 윈도우 버전
학습된 PyTorch 모델을 사용하여 마이크 입력을 실시간으로 분류합니다.
"""

import os
import time
import warnings
from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

import pyaudio
import serial

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================= CNN 모델 =========================
class SimpleCNN(nn.Module):
    """
    학습 시 사용한 모델과 동일한 CNN 구조

    4개의 합성곱 블록과 3개의 완전연결 레이어로 구성된 분류 모델입니다.
    각 블록은 합성곱, 배치정규화, ReLU 활성화, 풀링, 드롭아웃으로 구성됩니다.

    주의: 이 구조는 학습된 모델과 정확히 일치해야 합니다.
          레이어 구조나 파라미터를 변경하면 학습된 가중치를 로드할 수 없습니다.
    """

    def __init__(self, num_classes=10):
        """
        Args:
            num_classes (int): 분류할 클래스 개수 (기본값: 10)
                              UrbanSound8K 데이터셋 기준 10개 클래스
        """
        super().__init__()

        # ===== 특징 추출 레이어 (Convolutional layers) =====
        self.features = nn.Sequential(
            # Block 1: 첫 번째 합성곱 블록
            nn.Conv2d(3, 64, 3, padding=1),  # 입력 3채널 -> 64채널
            nn.BatchNorm2d(64),  # 배치 정규화 (학습 안정화)
            nn.ReLU(inplace=True),  # 활성화 함수
            nn.Conv2d(64, 64, 3, padding=1),  # 64채널 -> 64채널
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 크기 1/2로 축소
            nn.Dropout2d(0.1),  # 과적합 방지 (10% 드롭아웃)

            # Block 2: 두 번째 합성곱 블록
            nn.Conv2d(64, 128, 3, padding=1),  # 64채널 -> 128채널
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),  # 128채널 -> 128채널
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 크기 1/2로 축소
            nn.Dropout2d(0.2),  # 과적합 방지 (20% 드롭아웃)

            # Block 3: 세 번째 합성곱 블록
            nn.Conv2d(128, 256, 3, padding=1),  # 128채널 -> 256채널
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),  # 256채널 -> 256채널
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 크기 1/2로 축소
            nn.Dropout2d(0.2),

            # Block 4: 네 번째 합성곱 블록 (가장 깊은 레이어)
            nn.Conv2d(256, 512, 3, padding=1),  # 256채널 -> 512채널
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  # 512채널 -> 512채널
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 크기 1/2로 축소
            nn.Dropout2d(0.3),  # 과적합 방지 (30% 드롭아웃)
        )

        # ===== 적응형 평균 풀링 =====
        # 특징 맵 크기를 7x7로 고정 (입력 크기에 관계없이)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ===== 분류 레이어 (Fully connected layers) =====
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),  # 512*7*7 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 과적합 방지 (50% 드롭아웃)
            nn.Linear(1024, 512),  # 1024 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 512 -> 10 (최종 클래스 수)
        )

    def forward(self, x):
        """
        순전파 함수

        Args:
            x (torch.Tensor): 입력 텐서 (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: 클래스별 점수 (batch_size, num_classes)
        """
        x = self.features(x)  # 특징 추출
        x = self.avgpool(x)  # 적응형 풀링
        x = torch.flatten(x, 1)  # 1차원으로 펼침 (배치 차원 제외)
        x = self.classifier(x)  # 분류
        return x


# ========================= 실시간 분류기 =========================
class RealtimeAudioClassifier:
    """
    실시간 오디오 분류기 (슬라이딩 윈도우 + 예측 안정화 방식)

    주요 특징:
    - 슬라이딩 윈도우: 연속적으로 오디오를 수집하여 설정된 간격(기본 1초)마다 예측
    - 예측 안정화: 최근 3개의 예측 결과를 다수결로 결정하여 오탐 감소
    - 리샘플링: 44100Hz로 수집한 오디오를 22050Hz로 변환하여 모델 입력
    - 시리얼 통신: Arduino 등 외부 장치로 감지 결과 전송 가능
    """

    def __init__(self, model_path, target_labels=None,
                 serial_port=None, baud_rate=9600,
                 confidence_threshold=0.7, sliding_interval=1.0):
        """
        Args:
            model_path (str): 학습된 모델 파일 경로 (.pth 파일)
            target_labels (list): 감지할 타겟 라벨 리스트
                                 예시: ['car_horn', 'siren', 'engine_idling']
                                 빈 리스트면 모든 클래스 감지
            serial_port (str): 시리얼 포트 이름
                              Windows: 'COM3', 'COM4' 등
                              Linux: '/dev/ttyUSB0', '/dev/ttyACM0' 등
                              Mac: '/dev/cu.usbserial-1420' 등
                              None이면 시리얼 통신 비활성화
            baud_rate (int): 시리얼 통신 속도 (기본값: 9600)
                            일반적인 값: 9600, 19200, 38400, 57600, 115200
            confidence_threshold (float): 최소 신뢰도 임계값 (0.0~1.0)
                                         이 값 이상의 신뢰도를 가진 예측만 사용
                                         예시: 0.7 = 70% 이상의 신뢰도 필요
            sliding_interval (float): 슬라이딩 윈도우 간격 (초)
                                     이 간격마다 새로운 예측 수행
                                     예시: 1.0 = 1초마다 예측
                                          0.7 = 0.7초마다 예측 (더 빠른 반응)
        """
        self.device = device

        # ===== 학습 시 사용한 오디오 처리 설정 =====
        # 주의: 이 값들은 모델 학습 시 사용한 것과 동일해야 합니다.
        self.MODEL_SAMPLE_RATE = 22050  # 모델 학습 샘플레이트
        self.AUDIO_DURATION = 4.0  # 분석 오디오 길이 (초)
        self.N_MELS = 128  # 멜 스펙트로그램 주파수 빈 개수
        self.N_FFT = 2048  # FFT 윈도우 크기
        self.HOP_LENGTH = 512  # 홉 길이
        self.F_MIN = 20  # 최소 주파수 (Hz)
        self.F_MAX = 8000  # 최대 주파수 (Hz)
        self.SPEC_HEIGHT = 224  # 스펙트로그램 높이
        self.SPEC_WIDTH = 224  # 스펙트로그램 너비

        # ===== 실시간 수집 설정 =====
        self.MIC_SAMPLE_RATE = 44100  # 마이크 샘플링 레이트
        # 높은 샘플레이트로 수집 후 다운샘플링
        self.CHUNK = 1024  # 한 번에 읽을 프레임 수
        self.FORMAT = pyaudio.paFloat32  # 오디오 데이터 형식
        self.CHANNELS = 1  # 모노 채널

        # ===== 슬라이딩 윈도우 설정 =====
        self.sliding_interval = sliding_interval
        # 윈도우 샘플 수 (22050Hz 기준 4초)
        self.window_samples = int(self.MODEL_SAMPLE_RATE * self.AUDIO_DURATION)
        # 슬라이딩 샘플 수 (22050Hz 기준)
        self.slide_samples = int(self.MODEL_SAMPLE_RATE * sliding_interval)

        # ===== 신뢰도 임계값 =====
        # 이 값 이상의 신뢰도를 가진 예측만 안정화 버퍼에 추가
        self.confidence_threshold = confidence_threshold

        # ===== 클래스 이름 정의 =====
        # UrbanSound8K 데이터셋의 10개 클래스
        # 주의: 인덱스 순서는 학습 시 사용한 것과 동일해야 합니다.
        self.CLASS_NAMES = {
            0: "air_conditioner",  # 에어컨 소리
            1: "car_horn",  # 자동차 경적
            2: "children_playing",  # 어린이 놀이 소리
            3: "dog_bark",  # 개 짖는 소리
            4: "drilling",  # 드릴 소리
            5: "engine_idling",  # 엔진 공회전 소리
            6: "gun_shot",  # 총소리
            7: "jackhammer",  # 착암기 소리
            8: "siren",  # 사이렌 소리
            9: "street_music"  # 거리 음악
        }

        # ===== 타겟 라벨 설정 =====
        # 감지하고자 하는 특정 소리 클래스
        self.target_labels = target_labels if target_labels else []

        # ===== 예측 안정화 버퍼 =====
        # 최근 3개의 예측 결과를 저장하여 다수결로 최종 예측 결정
        # 오탐(false positive)을 줄이기 위한 메커니즘
        # maxlen=3: 최대 3개까지만 저장, 초과 시 가장 오래된 항목 자동 제거
        self.prediction_buffer = deque(maxlen=3)

        # ===== 오디오 데이터 버퍼 =====
        # 슬라이딩 윈도우를 위한 연속 오디오 데이터 저장
        # 44100Hz 샘플레이트로 수집된 원본 데이터
        self.audio_buffer = np.array([], dtype=np.float32)

        # ===== 초기화 정보 출력 =====
        print(f'\n{"=" * 70}')
        print(f'실시간 오디오 분류기 초기화')
        print(f'{"=" * 70}')
        print(f'Device: {self.device}')
        print(f'마이크 샘플레이트: {self.MIC_SAMPLE_RATE} Hz')
        print(f'모델 샘플레이트: {self.MODEL_SAMPLE_RATE} Hz')
        print(f'오디오 길이: {self.AUDIO_DURATION} 초')
        print(f'슬라이딩 간격: {self.sliding_interval} 초')
        print(f'신뢰도 임계값: {self.confidence_threshold * 100:.0f}%')
        print(f'타겟 라벨: {self.target_labels if self.target_labels else "모든 클래스"}')

        # ===== 모델 로드 =====
        self._load_model(model_path)

        # ===== 리샘플러 초기화 =====
        # 44100Hz -> 22050Hz 변환을 위한 리샘플러
        # 마이크로 수집한 고해상도 오디오를 모델 입력에 맞게 다운샘플링
        self.resampler = T.Resample(
            orig_freq=self.MIC_SAMPLE_RATE,
            new_freq=self.MODEL_SAMPLE_RATE
        )

        # ===== 멜 스펙트로그램 변환 초기화 =====
        # 오디오 파형을 멜 스펙트로그램으로 변환하는 객체
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.MODEL_SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            f_min=self.F_MIN,
            f_max=self.F_MAX
        )

        # ===== 시리얼 포트 초기화 =====
        self.serial_conn = None
        if serial_port:
            # 시리얼 포트가 지정된 경우 연결 시도
            try:
                self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)
                time.sleep(2)  # 연결 안정화 대기
                print(f'시리얼 포트 연결: {serial_port} @ {baud_rate} bps')
            except Exception as e:
                print(f'시리얼 포트 연결 실패: {e}')
                self.serial_conn = None
        else:
            print(f'시리얼 포트: 미연결')

        print(f'{"=" * 70}\n')

    def _load_model(self, model_path):
        """
        학습된 모델 로드

        Args:
            model_path (str): 모델 체크포인트 파일 경로
        """
        print(f'모델 로딩 중: {model_path}')

        # 체크포인트 로드
        # map_location: GPU에서 학습한 모델을 CPU에서도 로드 가능
        checkpoint = torch.load(model_path, map_location=self.device)

        # 모델 인스턴스 생성
        self.model = SimpleCNN(num_classes=10)

        # 학습된 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 지정된 디바이스로 이동
        self.model.to(self.device)

        # 평가 모드 설정 (드롭아웃, 배치정규화 등이 추론 모드로 동작)
        self.model.eval()

        print(f'모델 로드 완료 (정확도: {checkpoint["best_acc"]:.2f}%)')

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
            bool: 전송 성공 여부
        """
        if self.serial_conn and self.serial_conn.is_open:
            try:
                # CSV 형식 메시지 생성
                message = f"{class_id},{class_name},{confidence:.2f}\n"

                # 시리얼 포트로 전송 (문자열을 바이트로 인코딩)
                self.serial_conn.write(message.encode())

                print(f'   [Serial TX: {message.strip()}]')
                return True
            except Exception as e:
                print(f'   [Serial TX failed: {e}]')
                return False
        else:
            print(f'   [Serial TX: Not connected]')
            return False

    def preprocess_audio(self, audio_data):
        """
        오디오 데이터를 모델 입력 형식으로 전처리

        처리 순서:
        1. NumPy 배열을 PyTorch 텐서로 변환
        2. 오디오 길이를 정확히 4초로 맞춤 (자르거나 패딩)
        3. 멜 스펙트로그램 생성 (시간-주파수 변환)
        4. dB 스케일로 변환 (사람의 청각 특성 반영)
        5. 정규화 (평균 0, 표준편차 1)
        6. 크기를 224x224로 조정
        7. 3채널로 복제 (CNN 입력 형식)

        Args:
            audio_data (numpy.ndarray): 원본 오디오 데이터 (1D 배열, 22050Hz)

        Returns:
            torch.Tensor: 전처리된 텐서 (1, 3, 224, 224)
        """
        # ===== NumPy -> PyTorch 텐서 변환 =====
        # unsqueeze(0): 채널 차원 추가 (1, length)
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

        # ===== 오디오 길이 조정 =====
        # 모델은 항상 정확히 4초 길이의 오디오를 기대
        target_length = int(self.MODEL_SAMPLE_RATE * self.AUDIO_DURATION)

        if waveform.shape[1] > target_length:
            # 4초보다 길면 중앙 부분 추출
            start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        else:
            # 4초보다 짧으면 0으로 패딩
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        # ===== 멜 스펙트로그램 생성 =====
        # 시간 도메인 -> 시간-주파수 도메인 변환
        mel_spec = self.mel_transform(waveform)

        # dB 스케일로 변환
        # 사람의 청각은 소리 강도를 로그 스케일로 인지
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # ===== 정규화 =====
        # 평균 0, 표준편차 1로 정규화
        # 학습 시와 동일한 분포를 유지하여 성능 향상
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)  # 1e-8: 0 나누기 방지

        # ===== 크기 조정 =====
        # 모델 입력 크기인 224x224로 리사이즈
        mel_spec_db = F.interpolate(
            mel_spec_db.unsqueeze(0),  # 배치 차원 추가
            size=(self.SPEC_HEIGHT, self.SPEC_WIDTH),
            mode='bilinear',  # 이중선형 보간
            align_corners=False
        ).squeeze(0)  # 배치 차원 제거

        # ===== 3채널로 변환 =====
        # CNN 모델이 RGB 이미지 형식(3채널)을 기대하므로
        # 단일 채널 스펙트로그램을 3번 복제
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        # 배치 차원 추가 후 반환 (1, 3, 224, 224)
        return mel_spec_db.unsqueeze(0)

    def predict(self, audio_data):
        """
        오디오 데이터 예측

        Args:
            audio_data (numpy.ndarray): 원본 오디오 데이터 (22050Hz)

        Returns:
            tuple: (predicted_class, confidence, probabilities)
                - predicted_class (int): 예측된 클래스 ID (0~9)
                - confidence (float): 예측 신뢰도 (0.0~1.0)
                - probabilities (torch.Tensor): 모든 클래스의 확률 분포 (10,)
        """
        # 전처리
        input_tensor = self.preprocess_audio(audio_data).to(self.device)

        # ===== 모델 추론 =====
        with torch.no_grad():  # 그래디언트 계산 비활성화
            # 순전파
            outputs = self.model(input_tensor)

            # Softmax로 확률로 변환
            probabilities = F.softmax(outputs, dim=1)[0]

            # 가장 높은 확률의 클래스 선택
            predicted_class = torch.argmax(probabilities).item()

            # 해당 클래스의 신뢰도
            confidence = probabilities[predicted_class].item()

        return predicted_class, confidence, probabilities

    def get_stable_prediction(self, predicted_class, confidence):
        """
        최근 3개 예측의 다수결로 안정화된 예측 반환

        예측 안정화 과정:
        1. 신뢰도가 임계값 이상인 예측만 버퍼에 추가
        2. 최근 3개의 예측 중 가장 많이 나타난 클래스 선택 (다수결)
        3. 해당 클래스의 평균 신뢰도 계산

        이를 통해 일시적인 오탐(false positive)을 줄이고
        일관된 예측 결과만 사용합니다.

        Args:
            predicted_class (int): 현재 예측된 클래스 ID
            confidence (float): 현재 예측의 신뢰도 (0.0~1.0)

        Returns:
            tuple or None: (안정화된 클래스, 평균 신뢰도) 또는 None
                          - 신뢰도 미달이거나 버퍼가 충분하지 않으면 None
                          - 그 외에는 (most_common_class, avg_confidence) 반환
        """
        # ===== 신뢰도 임계값 체크 =====
        # 신뢰도가 낮은 예측은 버퍼에 추가하지 않음
        if confidence < self.confidence_threshold:
            return None

        # ===== 버퍼에 추가 =====
        # 튜플 형태로 (클래스, 신뢰도) 저장
        # maxlen=3이므로 3개 초과 시 가장 오래된 것 자동 제거
        self.prediction_buffer.append((predicted_class, confidence))

        # ===== 버퍼 충분성 체크 =====
        # 최소 2개 이상의 예측이 있어야 다수결 가능
        if len(self.prediction_buffer) < 2:
            return None

        # ===== 클래스와 신뢰도 추출 =====
        classes = [pred[0] for pred in self.prediction_buffer]
        confidences = [pred[1] for pred in self.prediction_buffer]

        # ===== 다수결 =====
        # Counter로 각 클래스의 출현 횟수 계산
        # most_common(1): 가장 많이 나타난 클래스 1개 선택
        # [0][0]: (클래스, 횟수) 튜플에서 클래스만 추출
        most_common_class = Counter(classes).most_common(1)[0][0]

        # ===== 평균 신뢰도 계산 =====
        # 다수결로 선택된 클래스의 신뢰도들만 추출
        class_confidences = [conf for cls, conf in self.prediction_buffer
                             if cls == most_common_class]
        # 평균 계산
        avg_confidence = np.mean(class_confidences)

        return most_common_class, avg_confidence

    def start_realtime_classification(self):
        """
        실시간 분류 시작 (슬라이딩 윈도우 방식)

        동작 방식:
        1. 초기에 4초 분량의 오디오 버퍼를 채움
        2. 슬라이딩 간격(기본 1초)마다 다음 작업 수행:
           - 새로운 청크 읽어서 버퍼에 추가
           - 버퍼의 마지막 4초 추출
           - 44100Hz -> 22050Hz 리샘플링
           - 모델로 예측
           - 예측 안정화 (다수결)
           - 결과 출력 및 시리얼 전송
           - 버퍼에서 슬라이딩 간격만큼 제거
        3. Ctrl+C를 누르면 종료

        장점:
        - 연속적인 오디오 처리로 빠른 반응 속도
        - 예측 안정화로 오탐 감소
        - 슬라이딩 간격 조정으로 성능과 속도 균형 조절 가능
        """
        # PyAudio 인스턴스 생성
        p = pyaudio.PyAudio()

        # ===== 오디오 스트림 열기 =====
        try:
            stream = p.open(
                format=self.FORMAT,  # Float32 형식
                channels=self.CHANNELS,  # 모노 채널
                rate=self.MIC_SAMPLE_RATE,  # 44100Hz
                input=True,  # 입력 스트림
                frames_per_buffer=self.CHUNK  # 버퍼 크기 1024
            )
        except Exception as e:
            print(f'\n오디오 스트림 열기 실패: {e}')
            p.terminate()
            return

        print(f'{"=" * 70}')
        print(f'실시간 오디오 분류 시작')
        print(f'{"=" * 70}')
        print(f'[Ctrl+C로 종료]\n')

        try:
            # ===== 초기 버퍼 채우기 (4초 분량) =====
            # 44100Hz 기준 4초 샘플 수 계산
            initial_samples = int(self.MIC_SAMPLE_RATE * self.AUDIO_DURATION)
            # 필요한 청크 수 계산
            chunks_needed = int(initial_samples / self.CHUNK)

            # 청크 단위로 읽어서 버퍼에 추가
            for _ in range(chunks_needed):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

            # ===== 슬라이딩 윈도우 루프 =====
            while True:
                # 새로운 청크 읽기
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # 버퍼에 추가
                self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

                # ===== 슬라이딩 간격 체크 =====
                # 44100Hz 기준 슬라이딩 간격의 샘플 수
                # 예: 1초 = 44100 샘플
                mic_slide_samples = int(self.MIC_SAMPLE_RATE * self.sliding_interval)

                # 슬라이딩 간격만큼 샘플이 쌓였는지 확인
                if len(self.audio_buffer) >= initial_samples + mic_slide_samples:
                    # ===== 마지막 4초 추출 =====
                    # 버퍼의 끝에서부터 4초 분량 추출
                    audio_window = self.audio_buffer[-initial_samples:]

                    # ===== 44100Hz → 22050Hz 리샘플링 =====
                    # PyTorch 텐서로 변환
                    audio_tensor = torch.from_numpy(audio_window).float().unsqueeze(0)
                    # 리샘플러 적용
                    audio_resampled = self.resampler(audio_tensor).squeeze(0).numpy()

                    # ===== 예측 수행 =====
                    predicted_class, confidence, probabilities = self.predict(audio_resampled)

                    # ===== 안정화된 예측 얻기 =====
                    # 최근 3개의 예측 중 다수결로 결정
                    stable_result = self.get_stable_prediction(predicted_class, confidence)

                    if stable_result:
                        # 안정화된 예측이 있는 경우
                        stable_class, stable_confidence = stable_result
                        class_name = self.CLASS_NAMES[stable_class]

                        # ===== 타겟 라벨 감지 확인 =====
                        is_detected = (class_name in self.target_labels)

                        # ===== 결과 출력 =====
                        print(f'{"=" * 70}')

                        if is_detected:
                            # 타겟 감지됨
                            print(f'[DETECTED: {class_name.upper()} (Confidence: {stable_confidence * 100:.1f}%)]')

                            # 시리얼 전송
                            self.send_to_serial(stable_class, class_name, stable_confidence * 100)
                        else:
                            # 타겟 아님
                            if self.target_labels:
                                # 타겟 라벨이 설정되어 있는 경우
                                print(f'[DETECTED: NONE]')
                                print(f'   (Classified as: {class_name}, Confidence: {stable_confidence * 100:.1f}%)')

                                # 시리얼 연결 상태 표시
                                if self.serial_conn and self.serial_conn.is_open:
                                    print(f'   [Serial TX: Not sent (not target label)]')
                                else:
                                    print(f'   [Serial TX: Not connected]')
                            else:
                                # 타겟 라벨이 없는 경우 (모든 클래스 감지)
                                print(f'[DETECTED: {class_name.upper()} (Confidence: {stable_confidence * 100:.1f}%)]')
                                self.send_to_serial(stable_class, class_name, stable_confidence * 100)

                        print(f'{"=" * 70}\n')

                    # ===== 버퍼에서 슬라이딩 간격만큼 제거 =====
                    # 오래된 데이터 제거하여 버퍼 크기 유지
                    # 슬라이딩 윈도우의 핵심 부분
                    self.audio_buffer = self.audio_buffer[mic_slide_samples:]

        except KeyboardInterrupt:
            # Ctrl+C를 누르면 종료
            print(f'\n\n사용자에 의해 중지됨')

        except Exception as e:
            # 기타 오류 발생 시
            print(f'\n\n오류 발생: {e}')
            import traceback
            traceback.print_exc()

        finally:
            # ===== 정리 작업 =====
            print(f'\n정리 중...')

            # 오디오 스트림 종료
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 시리얼 포트 종료
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                print(f'시리얼 포트 종료')

            print(f'종료 완료')


# ========================= 메인 함수 =========================
def main():
    """
    메인 함수

    실시간 오디오 분류기를 설정하고 실행합니다.

    ===== 사용자 설정 가이드 =====

    1. 모델 경로 변경:
       model_path 변수를 실제 모델 파일 경로로 변경하세요.

    2. 감지할 소리 변경 (target_labels):
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
       target_labels = ['car_horn', 'siren']              # 경적과 사이렌만
       target_labels = ['dog_bark']                       # 개 짖는 소리만
       target_labels = []                                  # 모든 소리 감지

    3. 신뢰도 임계값 변경 (confidence_threshold):
       예측을 신뢰하기 위한 최소 신뢰도를 설정하세요.

       예시:
       confidence_threshold = 0.5    # 50% 이상 (더 많은 감지, 오탐 가능)
       confidence_threshold = 0.7    # 70% 이상 (기본값, 균형)
       confidence_threshold = 0.9    # 90% 이상 (정확도 중시, 놓칠 가능성)

    4. 슬라이딩 간격 변경 (sliding_interval):
       예측을 수행하는 간격을 설정하세요.

       예시:
       sliding_interval = 0.5    # 0.5초마다 예측 (빠른 반응, CPU 부하 증가)
       sliding_interval = 0.7    # 0.7초마다 예측 (권장)
       sliding_interval = 1.0    # 1초마다 예측 (기본값, 안정적)
       sliding_interval = 2.0    # 2초마다 예측 (느린 반응, CPU 부하 감소)

    5. 시리얼 포트 설정 (serial_port):
       실제 연결된 포트명으로 변경하세요.

       Windows 예시:
       serial_port = 'COM3'         # 장치 관리자에서 확인
       serial_port = 'COM4'

       Linux 예시:
       serial_port = '/dev/ttyUSB0'  # dmesg | grep tty 로 확인
       serial_port = '/dev/ttyACM0'

       Mac 예시:
       serial_port = '/dev/cu.usbserial-1420'
       serial_port = '/dev/cu.usbmodem14201'

       시리얼 통신 사용 안 함:
       serial_port = None

    6. 시리얼 통신 속도 변경 (baud_rate):
       연결된 장치와 동일한 속도로 설정하세요.

       일반적인 값:
       baud_rate = 9600      # 가장 일반적 (Arduino 기본값)
       baud_rate = 19200
       baud_rate = 38400
       baud_rate = 57600
       baud_rate = 115200    # 고속 통신
    """
    # ===== 모델 경로 설정 =====
    # 학습된 모델 파일의 경로를 지정하세요.
    model_path = './saved_models/vehicle_audio_balanced_with_sonyc_best.pth'

    # ===== 모델 파일 존재 확인 =====
    if not os.path.exists(model_path):
        print(f'\n오류: 모델 파일을 찾을 수 없습니다: {model_path}')
        print(f'모델을 먼저 학습시켜주세요.\n')
        return

    # ===== 타겟 라벨 설정 =====
    # 감지하고 싶은 소리를 리스트로 지정하세요.
    # 차량 관련 소리 3가지를 감지합니다.
    target_labels = ['car_horn', 'engine_idling', 'siren']

    # 다른 예시:
    # target_labels = ['dog_bark', 'children_playing']  # 개와 어린이 소리
    # target_labels = ['siren']                         # 사이렌만
    # target_labels = []                                # 모든 클래스

    # ===== 시리얼 포트 설정 =====
    # 실제 포트명으로 변경하세요.
    # 현재는 None으로 시리얼 통신 비활성화됨.
    serial_port = None  # 시리얼 통신 사용 안 함

    # 실제 사용 예시 (주석 제거 후 사용):
    # serial_port = 'COM3'                    # Windows
    # serial_port = '/dev/ttyUSB0'            # Linux
    # serial_port = '/dev/cu.usbserial-1420'  # Mac

    # ===== 시리얼 통신 속도 설정 =====
    baud_rate = 9600

    # 다른 속도 예시:
    # baud_rate = 115200  # 고속 통신

    # ===== 조정 가능한 파라미터 =====
    # 신뢰도 임계값 (0.0 ~ 1.0)
    # 이 값 이상의 신뢰도를 가진 예측만 사용
    confidence_threshold = 0.5

    # 슬라이딩 간격 (초)
    # 이 간격마다 새로운 예측 수행
    sliding_interval = 0.7

    # ===== 분류기 생성 및 시작 =====
    classifier = RealtimeAudioClassifier(
        model_path=model_path,
        target_labels=target_labels,
        serial_port=serial_port,
        baud_rate=baud_rate,
        confidence_threshold=confidence_threshold,
        sliding_interval=sliding_interval
    )

    # 실시간 분류 시작
    classifier.start_realtime_classification()


if __name__ == '__main__':
    main()