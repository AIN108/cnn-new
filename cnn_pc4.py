"""
실시간 차량 소리 분류기 (슬라이딩 윈도우 + 예측 안정화)
학습된 PyTorch 모델을 사용하여 마이크 입력을 실시간으로 분류합니다.
"""

import os
import numpy as np
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from collections import deque, Counter
import time
import serial


# ========================= 설정 =========================
class Config:
    """모델 학습 시 사용한 설정과 동일하게 유지"""
    MODEL_SAMPLE_RATE = 22050
    MIC_SAMPLE_RATE = 44100
    AUDIO_DURATION = 4.0
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 20
    F_MAX = 8000
    SPEC_HEIGHT = 224
    SPEC_WIDTH = 224

    # 실시간 수집 설정
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1

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


config = Config()


# ========================= 모델 정의 =========================
class SimpleCNN(nn.Module):
    """학습 시 사용한 모델과 동일한 구조"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ========================= 실시간 분류기 =========================
class RealtimeAudioClassifier:
    """실시간 오디오 분류기 (슬라이딩 윈도우 + 예측 안정화)"""

    def __init__(self, model_path, target_labels=None, confidence_threshold=0.5,
                 sliding_interval=0.7, serial_port=None, baud_rate=9600):
        """
        Args:
            model_path: 학습된 모델 경로 (.pth 파일)
            target_labels: 감지할 타겟 라벨 리스트 (예: ['car_horn', 'siren'])
            confidence_threshold: 감지 임계값 (0~1)
            sliding_interval: 슬라이딩 윈도우 간격 (초)
            serial_port: 시리얼 포트 (예: 'COM3', '/dev/ttyUSB0')
            baud_rate: 시리얼 통신 속도 (기본: 9600)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_labels = target_labels or []
        self.confidence_threshold = confidence_threshold
        self.sliding_interval = sliding_interval

        # 오디오 데이터 버퍼
        self.audio_buffer = np.array([], dtype=np.float32)

        # 예측 안정화 버퍼 (최근 3개 예측 저장)
        self.prediction_buffer = deque(maxlen=3)

        # 모델 로드
        print(f"모델 로딩 중: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SimpleCNN(num_classes=10)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"모델 로드 완료")
        print(f"   Device: {self.device}")
        if 'best_acc' in checkpoint:
            print(f"   Training accuracy: {checkpoint['best_acc']:.2f}%")

        # 리샘플러 (44100Hz → 22050Hz)
        self.resampler = T.Resample(
            orig_freq=config.MIC_SAMPLE_RATE,
            new_freq=config.MODEL_SAMPLE_RATE
        )

        # 멜 스펙트로그램 변환
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.MODEL_SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )

        # PyAudio 설정
        self.p = pyaudio.PyAudio()
        self.stream = None

        # 시리얼 포트 설정
        self.serial_conn = None
        self.serial_enabled = False
        if serial_port:
            try:
                self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)
                time.sleep(2)
                self.serial_enabled = True
                print(f"시리얼 포트 연결: {serial_port} @ {baud_rate} bps")
            except Exception as e:
                print(f"시리얼 포트 연결 실패: {e}")
                self.serial_conn = None
                self.serial_enabled = False
        else:
            print("시리얼 포트: 미연결")

        # 통계
        self.detection_count = {label: 0 for label in self.target_labels}
        self.total_predictions = 0

    def send_to_serial(self, class_id, class_name, confidence):
        """
        시리얼로 데이터 전송

        Returns:
            str: 전송 상태 메시지
        """
        if not self.serial_enabled:
            return "NONE"

        if self.serial_conn and self.serial_conn.is_open:
            try:
                # 형식: "CLASS_ID,CLASS_NAME,CONFIDENCE\n"
                message = f"{class_id},{class_name},{confidence:.2f}\n"
                self.serial_conn.write(message.encode())
                return f"SENT: {message.strip()}"
            except Exception as e:
                return f"ERROR: {e}"
        else:
            return "DISCONNECTED"

    def preprocess_audio(self, audio_data):
        """오디오 데이터를 모델 입력 형식으로 전처리"""
        # numpy array를 torch tensor로 변환
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

        # 길이 조정 (정확히 4초)
        target_length = int(config.MODEL_SAMPLE_RATE * config.AUDIO_DURATION)
        if waveform.shape[1] > target_length:
            start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        else:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        # 멜 스펙트로그램 생성
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # 정규화
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)

        # 크기 조정
        if mel_spec_db.shape[-1] != config.SPEC_WIDTH or mel_spec_db.shape[-2] != config.SPEC_HEIGHT:
            mel_spec_db = F.interpolate(
                mel_spec_db.unsqueeze(0),
                size=(config.SPEC_HEIGHT, config.SPEC_WIDTH),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 3채널로 복제
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        return mel_spec_db.unsqueeze(0)

    def predict(self, audio_data):
        """오디오 데이터 분류"""
        # 전처리
        input_tensor = self.preprocess_audio(audio_data).to(self.device)

        # 추론
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        return predicted_class, confidence, probabilities

    def get_stable_prediction(self, predicted_class, confidence):
        """
        최근 3개 예측의 다수결로 안정화된 예측 반환

        Returns:
            tuple or None: (안정화된 클래스, 평균 신뢰도) 또는 None
        """
        # 신뢰도 임계값 미달 시 제외
        if confidence < self.confidence_threshold:
            return None

        # 버퍼에 추가
        self.prediction_buffer.append((predicted_class, confidence))

        # 버퍼가 충분히 차지 않았으면 None
        if len(self.prediction_buffer) < 2:
            return None

        # 최근 예측들의 클래스와 신뢰도 추출
        classes = [pred[0] for pred in self.prediction_buffer]
        confidences = [pred[1] for pred in self.prediction_buffer]

        # 다수결
        most_common_class = Counter(classes).most_common(1)[0][0]

        # 해당 클래스의 평균 신뢰도
        class_confidences = [conf for cls, conf in self.prediction_buffer if cls == most_common_class]
        avg_confidence = np.mean(class_confidences)

        return most_common_class, avg_confidence

    def start_realtime_detection(self):
        """실시간 감지 시작 (슬라이딩 윈도우 방식)"""
        print(f"\n{'=' * 70}")
        print(f"실시간 오디오 분류 시작")
        print(f"{'=' * 70}")
        print(f"마이크 샘플레이트: {config.MIC_SAMPLE_RATE} Hz")
        print(f"모델 샘플레이트: {config.MODEL_SAMPLE_RATE} Hz")
        print(f"오디오 길이: {config.AUDIO_DURATION} 초")
        print(f"슬라이딩 간격: {self.sliding_interval} 초")
        print(f"신뢰도 임계값: {self.confidence_threshold * 100:.0f}%")
        print(f"타겟 라벨: {self.target_labels if self.target_labels else '모든 클래스'}")
        print(f"시리얼 포트: {'연결됨' if self.serial_enabled else '미연결'}")
        print(f"{'=' * 70}\n")

        # 오디오 스트림 시작
        try:
            self.stream = self.p.open(
                format=config.FORMAT,
                channels=config.CHANNELS,
                rate=config.MIC_SAMPLE_RATE,
                input=True,
                frames_per_buffer=config.CHUNK
            )
        except Exception as e:
            print(f"오디오 스트림 열기 실패: {e}")
            self.p.terminate()
            return

        print("오디오 수집 시작... (Ctrl+C로 종료)\n")

        try:
            # 초기 버퍼 채우기 (4초)
            initial_samples = int(config.MIC_SAMPLE_RATE * config.AUDIO_DURATION)
            chunks_needed = int(initial_samples / config.CHUNK)

            print("초기 버퍼 채우는 중...", end=" ", flush=True)
            for _ in range(chunks_needed):
                data = self.stream.read(config.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
            print("완료\n")

            # 슬라이딩 윈도우 루프
            while True:
                # 새로운 청크 읽기
                data = self.stream.read(config.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # 버퍼에 추가
                self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

                # 슬라이딩 간격만큼 샘플이 쌓였는지 확인
                mic_slide_samples = int(config.MIC_SAMPLE_RATE * self.sliding_interval)

                if len(self.audio_buffer) >= initial_samples + mic_slide_samples:
                    # 마지막 4초 추출
                    audio_window = self.audio_buffer[-initial_samples:]

                    # 44100Hz → 22050Hz 리샘플링
                    audio_tensor = torch.from_numpy(audio_window).float().unsqueeze(0)
                    audio_resampled = self.resampler(audio_tensor).squeeze(0).numpy()

                    # 예측
                    predicted_class, confidence, probabilities = self.predict(audio_resampled)

                    # 안정화된 예측 얻기
                    stable_result = self.get_stable_prediction(predicted_class, confidence)

                    if stable_result:
                        stable_class, stable_confidence = stable_result
                        class_name = config.CLASS_NAMES[stable_class]

                        # 통계 업데이트
                        self.total_predictions += 1

                        # 타겟 라벨 감지 여부 확인
                        is_target_detected = (class_name in self.target_labels)

                        # 결과 출력
                        print(f"{'=' * 70}")

                        if is_target_detected:
                            # 타겟 감지됨
                            self.detection_count[class_name] += 1

                            print(f"[DETECTED: {class_name.upper()}]")
                            print(f"   신뢰도: {stable_confidence * 100:.1f}%")

                            # 시리얼 전송
                            serial_status = self.send_to_serial(stable_class, class_name, stable_confidence * 100)
                            print(f"   시리얼: {serial_status}")

                        else:
                            # 타겟 아님
                            if self.target_labels:
                                print(f"[DETECTED: NONE]")
                                print(f"   (분류 결과: {class_name}, 신뢰도: {stable_confidence * 100:.1f}%)")
                            else:
                                print(f"[DETECTED: {class_name.upper()}]")
                                print(f"   신뢰도: {stable_confidence * 100:.1f}%")
                                serial_status = self.send_to_serial(stable_class, class_name, stable_confidence * 100)
                                print(f"   시리얼: {serial_status}")

                        print(f"{'=' * 70}\n")

                    # 버퍼에서 슬라이딩 간격만큼 제거
                    self.audio_buffer = self.audio_buffer[mic_slide_samples:]

        except KeyboardInterrupt:
            print("\n\n사용자에 의해 중지됨")
            self.stop()
            self.print_statistics()

        except Exception as e:
            print(f"\n\n오류 발생: {e}")
            import traceback
            traceback.print_exc()
            self.stop()

    def stop(self):
        """스트림 종료"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("시리얼 포트 종료")

        print("오디오 스트림 종료")

    def print_statistics(self):
        """통계 출력"""
        print(f"\n{'=' * 70}")
        print(f"감지 통계")
        print(f"{'=' * 70}")
        print(f"총 예측 횟수: {self.total_predictions}")
        if self.target_labels:
            print(f"\n타겟 라벨 감지 횟수:")
            for label in self.target_labels:
                count = self.detection_count[label]
                percentage = (count / self.total_predictions * 100) if self.total_predictions > 0 else 0
                print(f"   {label:20s}: {count:4d} 회 ({percentage:5.2f}%)")
        print(f"{'=' * 70}\n")


# ========================= 메인 실행 =========================
def main():
    """메인 함수"""
    # 설정
    MODEL_PATH = './saved_models/vehicle_audio_simple_integration_best.pth'

    # 감지할 타겟 라벨 설정
    TARGET_LABELS = ['car_horn', 'engine_idling', 'siren']

    # 신뢰도 임계값 (0.0 ~ 1.0)
    CONFIDENCE_THRESHOLD = 0.5

    # 슬라이딩 윈도우 간격 (초)
    SLIDING_INTERVAL = 0.7

    # 시리얼 포트 설정
    # Windows: 'COM3', 'COM4' 등
    # Linux: '/dev/ttyUSB0', '/dev/ttyACM0' 등
    # Mac: '/dev/cu.usbserial-1420' 등
    SERIAL_PORT = None  # 실제 포트명으로 변경하세요
    BAUD_RATE = 9600

    # 모델 경로 확인
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print(f"올바른 경로를 설정해주세요.")
        return

    # 분류기 생성
    classifier = RealtimeAudioClassifier(
        model_path=MODEL_PATH,
        target_labels=TARGET_LABELS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sliding_interval=SLIDING_INTERVAL,
        serial_port=SERIAL_PORT,
        baud_rate=BAUD_RATE
    )

    # 실시간 감지 시작
    classifier.start_realtime_detection()


if __name__ == '__main__':
    main()