"""
실시간 오디오 분류기 - 개선된 버전
정확도 향상을 위한 다양한 전처리 및 필터링 기법 적용

주요 개선사항:
1. RMS 기반 오디오 정규화
2. 무음 감지 및 스킵
3. 다수결 투표 방식 (voting_window)
4. 신뢰도 임계값 필터링
5. 오디오 품질 통계 표시
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
from collections import deque, Counter


# ========================= SimpleCNN 모델 정의 =========================
class SimpleCNN(nn.Module):
    """학습에 사용한 것과 동일한 모델 구조"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

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
    """실시간 오디오 분류기 - 개선된 버전"""

    def __init__(self, model_path, target_labels=None, serial_port=None, baud_rate=9600,
                 confidence_threshold=0.6, voting_window=3, show_audio_stats=True):

        # 오디오 설정
        self.SAMPLE_RATE = 22050
        self.AUDIO_DURATION = 4.0
        self.N_MELS = 128
        self.N_FFT = 2048
        self.HOP_LENGTH = 512
        self.F_MIN = 20
        self.F_MAX = 8000
        self.SPEC_HEIGHT = 224
        self.SPEC_WIDTH = 224

        # 개선 설정
        self.confidence_threshold = confidence_threshold
        self.voting_window = voting_window
        self.show_audio_stats = show_audio_stats
        self.prediction_history = deque(maxlen=voting_window)

        # 클래스 이름
        self.CLASS_NAMES = {
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

        # 타겟 라벨
        self.target_labels = target_labels if target_labels else ['car_horn', 'engine_idling', 'siren']

        # 시리얼 통신
        self.serial_conn = None
        if serial_port:
            try:
                self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)
                print(f'[시리얼 포트 연결: {serial_port} @ {baud_rate}bps]')
            except Exception as e:
                print(f'[시리얼 포트 연결 실패: {e}]')
        else:
            print('\n[사용 가능한 시리얼 포트]')
            ports = serial.tools.list_ports.comports()
            for port in ports:
                print(f'  - {port.device}: {port.description}')

        # 디바이스
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n[Device: {self.device}]')

        # 모델 로드
        print(f'[모델 로딩: {model_path}]')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SimpleCNN(num_classes=10)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f'[모델 로드 완료 - 정확도: {checkpoint["best_acc"]:.2f}%]')

        # Mel Spectrogram 변환
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            f_min=self.F_MIN,
            f_max=self.F_MAX
        )

        # PyAudio
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1

        print(f'\n[설정]')
        print(f'  타겟 라벨: {self.target_labels}')
        print(f'  신뢰도 임계값: {self.confidence_threshold:.2f}')
        print(f'  투표 윈도우: {self.voting_window}')

    def normalize_audio(self, audio_data):
        """RMS 기반 오디오 정규화"""
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms > 1e-6:
            target_rms = 0.1
            audio_data = audio_data * (target_rms / rms)
        return np.clip(audio_data, -1.0, 1.0)

    def check_audio_quality(self, audio_data):
        """오디오 품질 체크"""
        rms = np.sqrt(np.mean(audio_data ** 2))
        peak = np.max(np.abs(audio_data))
        is_silent = rms < 0.01

        return {'rms': rms, 'peak': peak, 'is_silent': is_silent}

    def send_to_serial(self, class_id, class_name, confidence):
        """시리얼 전송"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                message = f"{class_id},{class_name},{confidence:.2f}\n"
                self.serial_conn.write(message.encode())
            except Exception as e:
                print(f'[시리얼 전송 실패: {e}]')

    def preprocess_audio(self, audio_data):
        """전처리 - 학습 시와 동일"""
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

        target_length = int(self.SAMPLE_RATE * self.AUDIO_DURATION)
        if waveform.shape[1] > target_length:
            start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        else:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        mel_spec = self.mel_transform(waveform)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)

        mel_spec_db = F.interpolate(
            mel_spec_db.unsqueeze(0),
            size=(self.SPEC_HEIGHT, self.SPEC_WIDTH),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        mel_spec_db = mel_spec_db.repeat(3, 1, 1)
        return mel_spec_db.unsqueeze(0)

    def predict(self, audio_data):
        """예측"""
        input_tensor = self.preprocess_audio(audio_data).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        return predicted_class, confidence

    def get_voted_prediction(self):
        """다수결 투표"""
        if len(self.prediction_history) == 0:
            return None, 0.0, 0.0

        class_counts = Counter([pred['class'] for pred in self.prediction_history])
        most_common_class, count = class_counts.most_common(1)[0]

        confidences = [pred['confidence'] for pred in self.prediction_history
                       if pred['class'] == most_common_class]
        avg_confidence = np.mean(confidences)
        vote_ratio = count / len(self.prediction_history)

        return most_common_class, avg_confidence, vote_ratio

    def start_realtime_classification(self):
        """실시간 분류 시작"""
        p = pyaudio.PyAudio()

        print(f'\n[사용 가능한 입력 디바이스]')
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")

        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except Exception as e:
            print(f'[오디오 스트림 열기 실패: {e}]')
            p.terminate()
            return

        print(f'\n{"=" * 70}')
        print(f'[실시간 오디오 분류 시작 - 개선된 버전]')
        print(f'{"=" * 70}')
        print(f'종료하려면 Ctrl+C를 누르세요.\n')

        try:
            while True:
                data_buffer = []
                samples_needed = int(self.SAMPLE_RATE * self.AUDIO_DURATION)
                chunks_needed = int(samples_needed / self.CHUNK)

                print('[오디오 수집 중...]', end='', flush=True)

                for i in range(chunks_needed):
                    try:
                        data = stream.read(self.CHUNK, exception_on_overflow=False)
                        data_buffer.append(data)
                    except Exception as e:
                        continue

                audio_data = b''.join(data_buffer)
                audio_data = np.frombuffer(audio_data, dtype=np.float32)

                audio_quality = self.check_audio_quality(audio_data)

                if audio_quality['is_silent']:
                    print(f'\r[무음 감지 - 스킵]' + ' ' * 50)
                    time.sleep(0.1)
                    continue

                audio_data = self.normalize_audio(audio_data)

                predicted_class, confidence = self.predict(audio_data)
                class_name = self.CLASS_NAMES[predicted_class]

                self.prediction_history.append({
                    'class': predicted_class,
                    'confidence': confidence
                })

                voted_class, voted_confidence, vote_ratio = self.get_voted_prediction()

                if voted_class is not None:
                    voted_class_name = self.CLASS_NAMES[voted_class]
                else:
                    voted_class_name = class_name
                    voted_confidence = confidence
                    vote_ratio = 1.0

                print(f'\r{"=" * 70}')

                if self.show_audio_stats:
                    print(f'[오디오] RMS: {audio_quality["rms"]:.4f}, Peak: {audio_quality["peak"]:.4f}')

                print(f'[현재 프레임] {class_name} ({confidence * 100:.2f}%)')

                if voted_confidence >= self.confidence_threshold:
                    print(f'[최종 결과] {voted_class_name} ({voted_confidence * 100:.2f}%, 투표: {vote_ratio * 100:.0f}%)')

                    if voted_class_name in self.target_labels:
                        print(f'*** DETECTED: {voted_class_name.upper()} ***')
                        self.send_to_serial(voted_class, voted_class_name, voted_confidence * 100)
                else:
                    print(f'[최종 결과] 신뢰도 부족 ({voted_confidence * 100:.2f}% < {self.confidence_threshold * 100:.0f}%)')

                print(f'{"=" * 70}\n')
                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f'\n\n[사용자에 의해 중지됨]')
        except Exception as e:
            print(f'\n\n[오류 발생: {e}]')
            import traceback
            traceback.print_exc()
        finally:
            print(f'[정리 중...]')
            stream.stop_stream()
            stream.close()
            p.terminate()
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            print(f'[종료 완료]')


# ========================= 메인 함수 =========================
def main():
    model_path = './saved_models/vehicle_audio_balanced_with_sonyc_best.pth'

    if not os.path.exists(model_path):
        print(f'[오류] 모델 파일을 찾을 수 없습니다: {model_path}')
        return

    target_labels = ['car_horn', 'engine_idling', 'siren']
    serial_port = None  # 예: 'COM3'
    baud_rate = 9600

    confidence_threshold = 0.6  # 신뢰도 임계값
    voting_window = 3  # 투표 윈도우
    show_audio_stats = True  # 오디오 통계

    classifier = RealtimeAudioClassifier(
        model_path=model_path,
        target_labels=target_labels,
        serial_port=serial_port,
        baud_rate=baud_rate,
        confidence_threshold=confidence_threshold,
        voting_window=voting_window,
        show_audio_stats=show_audio_stats
    )

    classifier.start_realtime_classification()


if __name__ == '__main__':
    main()