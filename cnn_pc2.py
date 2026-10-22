"""
실시간 차량 소리 분류기 (PyTorch 버전)
학습된 VehicleSoundCNN 모델을 사용하여 실시간 오디오를 분류합니다.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from collections import deque

warnings.filterwarnings('ignore')


# ========================= 설정 =========================
class Config:
    """모델 설정 (학습 시와 동일하게 유지)"""
    # 오디오 설정
    SAMPLE_RATE = 22050
    AUDIO_DURATION = 4.0
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 20
    F_MAX = 8000

    # 스펙트로그램 크기
    SPEC_HEIGHT = 224
    SPEC_WIDTH = 224

    # 모델 경로
    MODEL_PATH = './saved_models/vehicle_audio_classifier_fsd50k_best.pth'

    # 클래스 이름
    CLASS_NAMES = {
        0: "air_conditioner",
        1: "car_horn",
        2: "children_playing",
        3: "dog_bark",
        4: "drilling",
        5: "engine_idling",
        6: "gun_shot",
        7: "jackhammer",
        8: "siren",
        9: "street_music"
    }

    # 한글 클래스 이름
    CLASS_NAMES_KR = {
        0: "에어컨",
        1: "자동차 경적",
        2: "아이들 노는 소리",
        3: "개 짖는 소리",
        4: "드릴 소리",
        5: "엔진 공회전",
        6: "총소리",
        7: "착암기",
        8: "사이렌",
        9: "거리 음악"
    }

    # 타겟 라벨 (차량 관련 소리)
    TARGET_LABELS = [1, 5, 8]  # car_horn, engine_idling, siren

    # 실시간 수집 설정
    CHUNK = 1024
    PYAUDIO_FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    PYAUDIO_RATE = 44100

    # 예측 임계값
    CONFIDENCE_THRESHOLD = 0.3  # 30% 이상일 때만 출력


config = Config()

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================= 모델 정의 =========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        se_weight = self.se(out)
        out = out * se_weight

        out += self.skip(x)
        out = self.relu(out)
        return out


class VehicleSoundCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.vehicle_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        features = x.view(x.size(0), -1)

        output = self.classifier(features)
        return output


# ========================= 전처리 클래스 =========================
class AudioPreprocessor:
    """오디오 전처리 (학습 시와 동일)"""

    def __init__(self, config):
        self.config = config

        # 멜 스펙트로그램 변환기
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )

        # 리샘플러 (44100 -> 22050)
        self.resampler = T.Resample(
            orig_freq=config.PYAUDIO_RATE,
            new_freq=config.SAMPLE_RATE
        )

    def preprocess(self, audio_data):
        """
        오디오 데이터를 모델 입력 형식으로 변환

        Args:
            audio_data: numpy array (raw audio)

        Returns:
            torch.Tensor: (1, 3, 224, 224) 크기의 멜 스펙트로그램
        """
        # numpy -> torch
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

        # 리샘플링 (44100 -> 22050)
        waveform = self.resampler(waveform)

        # 길이 조정 (4초 = 88200 샘플)
        target_length = int(self.config.SAMPLE_RATE * self.config.AUDIO_DURATION)

        if waveform.shape[1] > target_length:
            # 중앙 부분 추출
            start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        elif waveform.shape[1] < target_length:
            # 패딩
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        # 멜 스펙트로그램 생성
        mel_spec = self.mel_transform(waveform)

        # dB 스케일 변환
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # 정규화
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / std

        # 크기 조정 (128, X) -> (224, 224)
        mel_spec_db = F.interpolate(
            mel_spec_db.unsqueeze(0),
            size=(self.config.SPEC_HEIGHT, self.config.SPEC_WIDTH),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # 3채널로 복제 (grayscale -> RGB)
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        # 배치 차원 추가
        mel_spec_db = mel_spec_db.unsqueeze(0)

        return mel_spec_db


# ========================= 실시간 분류기 =========================
class RealtimeClassifier:
    """실시간 오디오 분류기"""

    def __init__(self, model_path, config):
        self.config = config
        self.device = device

        # 모델 로드
        print(f"🔄 모델 로딩 중: {model_path}")
        self.model = self._load_model(model_path)
        print(f"✅ 모델 로드 완료 (Device: {self.device})")

        # 전처리기 초기화
        self.preprocessor = AudioPreprocessor(config)

        # 예측 히스토리 (스무딩용)
        self.prediction_history = deque(maxlen=3)

        # PyAudio 초기화
        self.p = pyaudio.PyAudio()
        self.stream = None

        # 통계
        self.total_predictions = 0
        self.target_detections = 0

    def _load_model(self, model_path):
        """학습된 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # 모델 초기화
        model = VehicleSoundCNN(num_classes=10)

        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)  # config/history 포함 체크포인트라 필요. 직접 학습한 파일만 로드할 것

        # state_dict 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   📊 학습 정확도: {checkpoint.get('accuracy', 0):.2f}%")
            print(f"   🚗 차량 정확도: {checkpoint.get('vehicle_accuracy', 0):.2f}%")
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        return model

    def _collect_audio(self, duration):
        """지정된 시간만큼 오디오 수집"""
        frames = []
        num_chunks = int(self.config.PYAUDIO_RATE / self.config.CHUNK * duration)

        for _ in range(num_chunks):
            data = self.stream.read(self.config.CHUNK, exception_on_overflow=False)
            frames.append(data)

        # bytes -> numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        return audio_array

    def _predict(self, audio_data):
        """오디오 데이터에 대해 예측 수행"""
        with torch.no_grad():
            # 전처리
            input_tensor = self.preprocessor.preprocess(audio_data)
            input_tensor = input_tensor.to(self.device)

            # 예측
            output = self.model(input_tensor)

            # Softmax로 확률 변환
            probabilities = F.softmax(output, dim=1)[0].cpu().numpy()

            return probabilities

    def _smooth_predictions(self, probabilities):
        """예측 결과 스무딩 (최근 3개 평균)"""
        self.prediction_history.append(probabilities)

        if len(self.prediction_history) > 0:
            smoothed = np.mean(self.prediction_history, axis=0)
            return smoothed

        return probabilities

    def start(self):
        """실시간 분류 시작"""
        print("\n" + "=" * 70)
        print("🎤 실시간 차량 소리 분류기 시작")
        print("=" * 70)
        print(f"📊 샘플레이트: {self.config.PYAUDIO_RATE} Hz")
        print(f"⏱️  분석 간격: {self.config.AUDIO_DURATION:.1f}초")
        print(f"🚗 타겟 클래스: {', '.join([self.config.CLASS_NAMES[i] for i in self.config.TARGET_LABELS])}")
        print(f"🎯 신뢰도 임계값: {self.config.CONFIDENCE_THRESHOLD * 100:.0f}%")
        print("=" * 70)
        print("📢 Ctrl+C를 눌러 종료하세요\n")

        # 오디오 스트림 시작
        self.stream = self.p.open(
            format=self.config.PYAUDIO_FORMAT,
            channels=self.config.CHANNELS,
            rate=self.config.PYAUDIO_RATE,
            input=True,
            frames_per_buffer=self.config.CHUNK
        )

        try:
            while True:
                # 오디오 수집
                print('🔊 오디오 수집 중...', end=' ', flush=True)
                audio_data = self._collect_audio(self.config.AUDIO_DURATION)

                # 예측
                probabilities = self._predict(audio_data)

                # 스무딩
                smoothed_probs = self._smooth_predictions(probabilities)

                # 가장 높은 확률의 클래스
                predicted_class = np.argmax(smoothed_probs)
                confidence = smoothed_probs[predicted_class]

                self.total_predictions += 1

                # 결과 출력 (임계값 이상일 때만)
                if confidence >= self.config.CONFIDENCE_THRESHOLD:
                    class_name = self.config.CLASS_NAMES[predicted_class]
                    class_name_kr = self.config.CLASS_NAMES_KR[predicted_class]

                    # 타겟 클래스 감지 시 강조
                    if predicted_class in self.config.TARGET_LABELS:
                        self.target_detections += 1
                        print(f"\n🚨 [DETECTED] {class_name_kr} ({class_name}): {confidence * 100:.1f}%")
                        print(f"   ⚠️  차량 관련 소리 감지됨!")
                    else:
                        print(f"\n   {class_name_kr} ({class_name}): {confidence * 100:.1f}%")
                else:
                    print("(신뢰도 낮음)")

                # 상위 3개 클래스 출력
                top3_indices = np.argsort(smoothed_probs)[-3:][::-1]
                print("   [상위 3개]", end=" ")
                for idx in top3_indices:
                    print(f"{self.config.CLASS_NAMES_KR[idx]}({smoothed_probs[idx] * 100:.0f}%)", end=" ")
                print()

        except KeyboardInterrupt:
            print("\n\n⏹️  중지됨")
        finally:
            self._cleanup()

    def _cleanup(self):
        """리소스 정리"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.p.terminate()

        print("\n" + "=" * 70)
        print("📊 통계")
        print("=" * 70)
        print(f"   총 예측 횟수: {self.total_predictions}")
        print(f"   차량 소리 감지: {self.target_detections}회")
        if self.total_predictions > 0:
            detection_rate = (self.target_detections / self.total_predictions) * 100
            print(f"   감지율: {detection_rate:.1f}%")
        print("=" * 70)
        print("✅ 종료 완료")


# ========================= 메인 함수 =========================
def main():
    """메인 함수"""
    print("\n🚗 차량 소리 실시간 분류기 (PyTorch 버전)")
    print(f"🖥️  Device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 모델 경로 확인
    if not os.path.exists(config.MODEL_PATH):
        print(f"\n❌ 오류: 모델 파일을 찾을 수 없습니다")
        print(f"   경로: {config.MODEL_PATH}")
        print(f"\n💡 해결 방법:")
        print(f"   1. 모델을 먼저 학습시키세요")
        print(f"   2. 또는 MODEL_PATH를 올바른 경로로 수정하세요")
        return

    try:
        # 분류기 초기화 및 실행
        classifier = RealtimeClassifier(config.MODEL_PATH, config)
        classifier.start()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()