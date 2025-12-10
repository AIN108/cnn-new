"""
Ultimate Vehicle Sound Classification System v2.4 - HYBRID CORRECTED
⚖️ 어려운 클래스(Car Horn & Engine)에 집중!
✅ Car Horn & Engine: 실시간 정밀 분석
✅ Siren & 나머지: 캐싱 (이미 잘 되니까)
✅ 1 에폭: 15-20분 (2시간 대비 6-8배)
✅ 어려운 클래스 정확도 개선 목표!
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchaudio
import torchaudio.transforms as T
import librosa
import scipy.signal

from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {device}")

# ========================= Config =========================
class Config:
    URBANSOUND_PATH = r'C:\cnn\cnn_test\UrbanSound8K'
    ESC50_PATH = r'C:\cnn\cnn_test\ESC-50-master'
    FSD50K_PATH = r'C:\cnn\cnn_test\FSD50'
    SONYC_PATH = r'C:\cnn\cnn_test\SONYC'

    PRIMARY_TARGETS = [1, 5, 8]

    CLASSIFICATION_HIERARCHY = {
        'vehicle_sounds': [1, 5],
        'alert_sounds': [8],
        'mechanical_sounds': [4, 7],
        'human_sounds': [2, 9],
        'environmental_sounds': [0, 3, 6]
    }

    CLASS_TO_GROUP = {}
    for group, classes in CLASSIFICATION_HIERARCHY.items():
        for cls in classes:
            CLASS_TO_GROUP[cls] = group

    SAMPLE_RATE = 22050
    AUDIO_DURATION = 4.0
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512

    BATCH_SIZE = 24
    NUM_WORKERS = 4
    ACCUMULATION_STEPS = 3
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 300
    EARLY_STOPPING_PATIENCE = 30
    USE_AMP = True
    CLIP_GRAD_NORM = 0.5

    AUG_NOISE_PROB = 0.35
    AUG_PITCH_PROB = 0.25
    AUG_TIME_PROB = 0.25
    AUG_SPEC_PROB = 0.35

    # ⭐ 수정: 어려운 클래스에 집중!
    DIFFICULT_CLASSES = [1, 5]  # Car horn, Engine (실시간 정밀 분석)
    EASY_CLASSES = [8]  # Siren (캐싱 OK)

    SAVE_DIR = './ultimate_models_v24_hybrid_corrected'
    CACHE_DIR = './cache_v24_hybrid_corrected'
    MODEL_NAME = 'ultimate_vehicle_v24_hybrid_corrected'

    CLASS_NAMES = {
        0: "air_conditioner", 1: "car_horn", 2: "children_playing",
        3: "dog_bark", 4: "drilling", 5: "engine_idling",
        6: "gun_shot", 7: "jackhammer", 8: "siren", 9: "street_music"
    }


config = Config()
os.makedirs(config.SAVE_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)


# ========================= Enhanced Feature Extractor =========================
class EnhancedFeatureExtractor:
    """⭐ Car Horn과 Engine에 특화된 특징 추출"""

    def __init__(self):
        self.cache = {}

    def extract_essential_features(self, waveform_np):
        """31 essential features"""
        features = []

        rms = np.sqrt(np.mean(waveform_np**2))
        features.append(rms)

        zcr = np.mean(librosa.feature.zero_crossing_rate(waveform_np))
        features.append(zcr)

        centroid = np.mean(librosa.feature.spectral_centroid(y=waveform_np, sr=config.SAMPLE_RATE))
        features.append(centroid)

        mfcc = librosa.feature.mfcc(y=waveform_np, sr=config.SAMPLE_RATE, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        chroma = librosa.feature.chroma_stft(y=waveform_np, sr=config.SAMPLE_RATE)
        features.append(np.mean(chroma))
        features.append(np.std(chroma))

        return np.array(features, dtype=np.float32)

    def extract_car_horn_features(self, waveform_np):
        """⭐ Car Horn 정밀 분석"""
        features = []

        fft = np.fft.rfft(waveform_np)
        freqs = np.fft.rfftfreq(len(waveform_np), 1/config.SAMPLE_RATE)

        # 다중 주파수 대역 분석
        bands = [(800, 1200), (1200, 2000), (2000, 4000)]
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            energy = np.sum(np.abs(fft[mask])**2)
            features.append(energy)

        # 버스트 패턴 분석
        envelope = np.abs(scipy.signal.hilbert(waveform_np))
        threshold = np.mean(envelope) + 1.5 * np.std(envelope)
        bursts = envelope > threshold
        burst_changes = np.diff(bursts.astype(int))
        burst_count = len(np.where(burst_changes == 1)[0])
        features.append(burst_count)

        # 버스트 지속시간
        burst_starts = np.where(burst_changes == 1)[0]
        burst_ends = np.where(burst_changes == -1)[0]
        if len(burst_starts) > 0 and len(burst_ends) > 0:
            durations = [(burst_ends[i] - burst_starts[i]) / config.SAMPLE_RATE
                        for i in range(min(len(burst_starts), len(burst_ends)))]
            features.append(np.mean(durations))
        else:
            features.append(0)

        while len(features) < 5:
            features.append(0)

        return np.array(features[:5], dtype=np.float32)

    def extract_engine_features(self, waveform_np):
        """⭐ Engine 정밀 분석"""
        features = []

        fft = np.fft.rfft(waveform_np)
        freqs = np.fft.rfftfreq(len(waveform_np), 1/config.SAMPLE_RATE)

        # 저주파 대역 분석
        bands = [(30, 100), (100, 300), (300, 800)]
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            energy = np.sum(np.abs(fft[mask])**2)
            features.append(energy)

        # 주기성 분석
        autocorr = np.correlate(waveform_np, waveform_np, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        peaks, properties = scipy.signal.find_peaks(autocorr, height=0.2, distance=20)

        if len(peaks) > 0:
            fundamental = config.SAMPLE_RATE / peaks[0] if peaks[0] > 0 else 0
            features.append(fundamental)
            features.append(properties['peak_heights'][0])
        else:
            features.extend([0, 0])

        while len(features) < 5:
            features.append(0)

        return np.array(features[:5], dtype=np.float32)

    def extract_simple_features(self, waveform_np, target_class):
        """간단한 특징 (캐싱용)"""
        features = []

        if target_class == 1:  # car_horn 간단 버전
            fft = np.fft.rfft(waveform_np)
            freqs = np.fft.rfftfreq(len(waveform_np), 1/config.SAMPLE_RATE)
            high_mask = (freqs >= 1000) & (freqs <= 4000)
            features.append(np.sum(np.abs(fft[high_mask])**2))

            envelope = np.abs(scipy.signal.hilbert(waveform_np))
            threshold = np.mean(envelope) + 2 * np.std(envelope)
            bursts = envelope > threshold
            burst_changes = np.diff(bursts.astype(int))
            features.append(len(np.where(burst_changes == 1)[0]))

        elif target_class == 5:  # engine 간단 버전
            fft = np.fft.rfft(waveform_np)
            freqs = np.fft.rfftfreq(len(waveform_np), 1/config.SAMPLE_RATE)
            low_mask = (freqs >= 50) & (freqs <= 500)
            features.append(np.sum(np.abs(fft[low_mask])**2))

            autocorr = np.correlate(waveform_np, waveform_np, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            peaks = scipy.signal.find_peaks(autocorr, height=0.3)[0]
            periodicity = config.SAMPLE_RATE / peaks[0] if len(peaks) > 0 and peaks[0] > 0 else 0
            features.append(periodicity)

        elif target_class == 8:  # siren 간단 버전
            stft = np.abs(librosa.stft(waveform_np))
            features.append(np.var(stft, axis=1).mean())

            centroid = librosa.feature.spectral_centroid(y=waveform_np, sr=config.SAMPLE_RATE)
            features.append(np.mean(np.abs(np.diff(centroid.flatten()))))

        while len(features) < 5:
            features.append(0)

        return np.array(features[:5], dtype=np.float32)


# ========================= Dataset Mapper =========================
class DatasetMapper:
    @staticmethod
    def get_esc50_mapping():
        return {
            'car_horn': 1, 'engine': 5, 'siren': 8, 'dog': 3,
            'drilling': 4, 'breathing': 0, 'gunshot': 6, 'fireworks': 6,
            'jackhammer': 7, 'crying_baby': 2, 'laughing': 2, 'church_bells': 9,
        }

    @staticmethod
    def get_fsd50k_mapping():
        return {
            'Vehicle horn, car horn, honking': 1, 'Car horn': 1, 'Honking': 1, 'Horn': 1,
            'Engine': 5, 'Engine idling': 5, 'Idling': 5, 'Car engine': 5,
            'Vehicle engine': 5, 'Motor': 5,
            'Siren': 8, 'Emergency vehicle siren': 8, 'Police car (siren)': 8,
            'Ambulance (siren)': 8, 'Fire engine, fire truck (siren)': 8,
            'Air conditioning': 0, 'Air conditioner': 0, 'Fan': 0, 'Mechanical fan': 0,
            'Dog barking': 3, 'Bark': 3, 'Dog': 3, 'Bow-wow': 3,
            'Children playing': 2, 'Children shouting': 2, 'Baby cry, infant cry': 2,
            'Child speech, kid speaking': 2,
            'Drill': 4, 'Power drill': 4, 'Electric drill': 4,
            'Jackhammer': 7, 'Pneumatic drill': 7,
            'Gunshot, gunfire': 6, 'Explosion': 6, 'Machine gun': 6,
            'Street music': 9, 'Music': 9, 'Musical instrument': 9,
            'Guitar': 9, 'Drum': 9, 'Singing': 9,
        }

    @staticmethod
    def get_sonyc_mapping():
        return {
            '5-1_car-horn_presence': 1,
            '1-1_small-sounding-engine_presence': 5,
            '1-2_medium-sounding-engine_presence': 5,
            '1-3_large-sounding-engine_presence': 5,
            '1_engine_presence': 5,
            '5-3_siren_presence': 8,
            '8-1_dog-barking-whining_presence': 3,
            '8_dog_presence': 3,
            '7-1_person-or-small-group-talking_presence': 2,
            '7-2_person-or-small-group-shouting_presence': 2,
            '2-1_rock-drill_presence': 4,
            '4-2_small-medium-rotating-saw_presence': 4,
            '2-2_jackhammer_presence': 7,
            '2-3_hoe-ram_presence': 7,
            '6-1_stationary-music_presence': 9,
            '6-2_mobile-music_presence': 9,
            '6_music_presence': 9,
        }


# ========================= CorrectedHybrid Dataset =========================
class CorrectedHybridDataset(Dataset):
    """⭐ 수정: Car Horn & Engine에 집중!"""

    def __init__(self, is_training=True, fold=1):
        self.is_training = is_training
        self.fold = fold
        self.data = []
        self.feature_extractor = EnhancedFeatureExtractor()

        # 데이터 로드
        data_cache_file = Path(config.CACHE_DIR) / f"data_{'train' if is_training else 'val'}_f{fold}.pkl"

        if data_cache_file.exists():
            print("📦 Loading data cache...")
            with open(data_cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self._load_all_datasets()
            with open(data_cache_file, 'wb') as f:
                pickle.dump(self.data, f)

        if is_training:
            self._balance()

        # ⭐ 캐싱: Car Horn & Engine 제외!
        feature_cache_file = Path(config.CACHE_DIR) / f"features_{'train' if is_training else 'val'}_f{fold}.pkl"

        if feature_cache_file.exists():
            print("📦 Loading cached features (except car_horn & engine)...")
            with open(feature_cache_file, 'rb') as f:
                self.feature_cache = pickle.load(f)
            print(f"   ✅ Loaded {len(self.feature_cache)} cached features")
        else:
            print("🔧 Caching easy classes (siren & others)...")
            self._cache_easy_classes()
            with open(feature_cache_file, 'wb') as f:
                pickle.dump(self.feature_cache, f)
            print(f"   💾 Saved {len(self.feature_cache)} features")

        print(f"✅ Dataset: {len(self.data)} samples")
        self._stats()

    def _cache_easy_classes(self):
        """Car Horn & Engine 제외하고 캐싱"""
        self.feature_cache = {}

        for item in tqdm(self.data, desc='Caching'):
            path = item['path']
            label = item['label']

            # ⭐ Car Horn(1), Engine(5)은 캐싱 안 함!
            if label in config.DIFFICULT_CLASSES:
                continue

            if path in self.feature_cache:
                continue

            try:
                y, sr = librosa.load(path, sr=config.SAMPLE_RATE, duration=4.0)

                essential = self.feature_extractor.extract_essential_features(y)
                simple = self.feature_extractor.extract_simple_features(
                    y, label if label in config.PRIMARY_TARGETS else None
                )

                self.feature_cache[path] = {
                    'essential': essential,
                    'target': simple
                }
            except:
                self.feature_cache[path] = {
                    'essential': np.zeros(31, dtype=np.float32),
                    'target': np.zeros(5, dtype=np.float32)
                }

    def _load_all_datasets(self):
        """데이터 로드"""
        print("\n📦 Loading datasets...")

        # UrbanSound8K
        print("\n1. UrbanSound8K:")
        meta_path = Path(config.URBANSOUND_PATH) / 'metadata' / 'UrbanSound8K.csv'
        meta = pd.read_csv(meta_path)

        if self.is_training:
            meta = meta[meta['fold'] != self.fold]
        else:
            meta = meta[meta['fold'] == self.fold]

        urban_start = len(self.data)
        for _, row in tqdm(meta.iterrows(), total=len(meta), desc='  Loading'):
            path = Path(config.URBANSOUND_PATH) / 'audio' / f"fold{row['fold']}" / row['slice_file_name']
            if path.exists():
                self.data.append({'path': str(path), 'label': row['classID'], 'dataset': 'urbansound'})
        print(f"   ✅ {len(self.data) - urban_start} samples")

        # ESC-50
        print("\n2. ESC-50:")
        mapper = DatasetMapper()
        esc_mapping = mapper.get_esc50_mapping()
        esc_start = len(self.data)

        meta_file = Path(config.ESC50_PATH) / 'meta' / 'esc50.csv'
        if meta_file.exists():
            meta_df = pd.read_csv(meta_file)
            for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc='  Loading'):
                category = row['category']
                if category in esc_mapping:
                    audio_path = Path(config.ESC50_PATH) / 'audio' / row['filename']
                    if audio_path.exists():
                        self.data.append({
                            'path': str(audio_path),
                            'label': esc_mapping[category],
                            'dataset': 'esc50'
                        })
            print(f"   ✅ {len(self.data) - esc_start} samples")

        # FSD50K
        print("\n3. FSD50K:")
        fsd_mapping = mapper.get_fsd50k_mapping()
        fsd_start = len(self.data)

        for split, audio_dir in [('dev', 'FSD50K.dev_audio'), ('eval', 'FSD50K.eval_audio')]:
            meta = Path(config.FSD50K_PATH) / 'FSD50K.ground_truth' / f'{split}.csv'
            audio = Path(config.FSD50K_PATH) / audio_dir

            if not meta.exists() or not audio.exists():
                continue

            df = pd.read_csv(meta)
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f'  {split}'):
                fname = str(row['fname'])
                if not fname.endswith('.wav'):
                    fname = fname + '.wav'

                labels = str(row['labels']).split(',')
                for label in labels:
                    label = label.strip()
                    if label in fsd_mapping:
                        audio_path = audio / fname
                        if audio_path.exists():
                            try:
                                duration = librosa.get_duration(filename=str(audio_path))
                                if duration >= 0.5:
                                    self.data.append({
                                        'path': str(audio_path),
                                        'label': fsd_mapping[label],
                                        'dataset': f'fsd50k_{split}'
                                    })
                            except:
                                pass
                            break
        print(f"   ✅ {len(self.data) - fsd_start} samples")

        # SONYC
        print("\n4. SONYC:")
        sonyc_mapping = mapper.get_sonyc_mapping()
        sonyc_start = len(self.data)

        annotation_file = Path(config.SONYC_PATH) / 'annotations.csv'
        if annotation_file.exists():
            df = pd.read_csv(annotation_file)

            audio_files = {}
            for wav_file in Path(config.SONYC_PATH).glob('**/*.wav'):
                audio_files[wav_file.name] = str(wav_file)

            for _, row in tqdm(df.iterrows(), total=len(df), desc='  Loading'):
                filename = row['audio_filename']
                if filename not in audio_files:
                    continue

                for column_name, mapped_label in sonyc_mapping.items():
                    if column_name in df.columns and row[column_name] == 1:
                        self.data.append({
                            'path': audio_files[filename],
                            'label': mapped_label,
                            'dataset': 'sonyc'
                        })
                        break
            print(f"   ✅ {len(self.data) - sonyc_start} samples")

        print(f"\n📊 Total: {len(self.data)} samples")

    def _balance(self):
        """균형 조정"""
        print("\n⚖️ Balancing...")
        class_data = defaultdict(list)
        for item in self.data:
            class_data[item['label']].append(item)

        target_max = max([len(class_data[c]) for c in config.PRIMARY_TARGETS if class_data[c]])
        target_goal = min(target_max * 2, 10000)
        non_target_goal = int(target_goal * 0.3)

        balanced = []
        for i in range(10):
            samples = class_data[i]
            if not samples:
                continue

            goal = target_goal if i in config.PRIMARY_TARGETS else non_target_goal

            if len(samples) >= goal:
                selected = random.sample(samples, goal)
                balanced.extend([s.copy() for s in selected])
            else:
                n_repeats = goal // len(samples)
                remainder = goal % len(samples)

                for _ in range(n_repeats):
                    balanced.extend([s.copy() for s in samples])

                selected = random.sample(samples, remainder)
                balanced.extend([s.copy() for s in selected])

        self.data = balanced
        random.shuffle(self.data)

    def _stats(self):
        """통계"""
        counts = Counter([d['label'] for d in self.data])

        print("\n📊 Class distribution:")
        for i in range(10):
            count = counts.get(i, 0)
            pct = count / len(self.data) * 100 if self.data else 0
            mark = "🎯" if i in config.PRIMARY_TARGETS else "  "
            difficult = "⭐" if i in config.DIFFICULT_CLASSES else " "
            print(f"   {mark}{difficult} {config.CLASS_NAMES[i]:16s}: {count:5d} ({pct:4.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """⭐ 수정: Car Horn & Engine은 실시간!"""
        item = self.data[idx]

        # 오디오 로드
        waveform, sr = torchaudio.load(item['path'])

        if sr != config.SAMPLE_RATE:
            resampler = T.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        if self.is_training:
            waveform = self._augment(waveform)

        target_len = int(config.SAMPLE_RATE * config.AUDIO_DURATION)
        if waveform.shape[1] > target_len:
            start = random.randint(0, waveform.shape[1] - target_len) if self.is_training else 0
            waveform = waveform[:, start:start + target_len]
        else:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))

        # 멜 스펙트로그램
        mel_spec = self._compute_mel(waveform)

        # ⭐ 특징: Car Horn & Engine만 실시간 정밀 계산!
        waveform_np = waveform.squeeze().numpy()

        if item['label'] in config.DIFFICULT_CLASSES:
            # ⭐ Car Horn (1) or Engine (5): 정밀 분석
            essential = self.feature_extractor.extract_essential_features(waveform_np)

            if item['label'] == 1:  # Car horn
                target = self.feature_extractor.extract_car_horn_features(waveform_np)
            elif item['label'] == 5:  # Engine
                target = self.feature_extractor.extract_engine_features(waveform_np)
            else:
                target = np.zeros(5, dtype=np.float32)

            all_features = np.concatenate([essential, target])
        else:
            # ⚡ 나머지: 캐시
            cached = self.feature_cache.get(item['path'])
            if cached:
                all_features = np.concatenate([cached['essential'], cached['target']])
            else:
                all_features = np.zeros(36, dtype=np.float32)

        feature_tensor = torch.from_numpy(all_features).float()

        return {
            'spectrogram': mel_spec,
            'features': feature_tensor,
            'label': item['label']
        }

    def _augment(self, waveform):
        """증강"""
        if random.random() < config.AUG_NOISE_PROB:
            snr = random.uniform(15, 35)
            signal_power = torch.mean(waveform**2)
            noise_power = signal_power / (10**(snr/10))
            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
            waveform = waveform + noise

        if random.random() < config.AUG_PITCH_PROB:
            steps = random.choice([-2, -1, 1, 2])
            try:
                waveform_np = waveform.numpy().squeeze()
                shifted = librosa.effects.pitch_shift(waveform_np, sr=config.SAMPLE_RATE, n_steps=steps)
                waveform = torch.from_numpy(shifted).unsqueeze(0)
            except:
                pass

        return torch.clamp(waveform, -1.0, 1.0)

    def _compute_mel(self, waveform):
        """멜 스펙트로그램"""
        mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        )

        mel_spec = mel_transform(waveform)

        if self.is_training and random.random() < config.AUG_SPEC_PROB:
            time_mask = T.TimeMasking(time_mask_param=20)
            mel_spec = time_mask(mel_spec)

            freq_mask = T.FrequencyMasking(freq_mask_param=15)
            mel_spec = freq_mask(mel_spec)

        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-6)

        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        return mel_spec_db


# ========================= Model (동일) =========================

class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SpectralExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class TemporalExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class HierarchicalExpert(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2)
        )

        self.group_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

        self.class_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.features(x)
        group = self.group_cls(feat)
        cls = self.class_cls(feat)
        return cls, group


class FeatureExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(36, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class UltimateEnsemble(nn.Module):
    def __init__(self):
        super().__init__()

        self.spectral_expert = SpectralExpert()
        self.temporal_expert = TemporalExpert()
        self.hierarchical_expert = HierarchicalExpert()
        self.feature_expert = FeatureExpert()

        self.gate = nn.Sequential(
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )

        self.global_feat = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.meta = nn.Sequential(
            nn.Linear(44, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, spec, features):
        spec_out = self.spectral_expert(spec)
        temp_out = self.temporal_expert(spec)
        hier_out, group_out = self.hierarchical_expert(spec)
        feat_out = self.feature_expert(features)

        global_f = self.global_feat(spec)
        gate_weights = self.gate(global_f)

        expert_stack = torch.stack([spec_out, temp_out, hier_out, feat_out], dim=1)

        weighted = expert_stack * gate_weights.unsqueeze(-1)
        weighted_sum = weighted.sum(1)

        all_features = torch.cat([expert_stack.flatten(1), gate_weights], dim=1)
        meta_out = self.meta(all_features)

        final = weighted_sum * 0.6 + meta_out * 0.4

        return final, expert_stack, gate_weights


# ========================= Trainer =========================
class UltimateTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        params = [
            {'params': model.spectral_expert.parameters(), 'lr': config.LEARNING_RATE},
            {'params': model.temporal_expert.parameters(), 'lr': config.LEARNING_RATE},
            {'params': model.hierarchical_expert.parameters(), 'lr': config.LEARNING_RATE * 1.5},
            {'params': model.feature_expert.parameters(), 'lr': config.LEARNING_RATE * 0.5},
            {'params': model.gate.parameters(), 'lr': config.LEARNING_RATE * 0.1},
            {'params': model.meta.parameters(), 'lr': config.LEARNING_RATE * 0.1}
        ]

        self.optimizer = torch.optim.AdamW(params, weight_decay=2e-4)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[p['lr'] * 10 for p in params],
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
            pct_start=0.15,
            div_factor=25
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler = GradScaler() if config.USE_AMP else None

        self.best_acc = 0
        self.best_target_acc = 0
        self.patience = 0

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        corrects = 0
        total = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:3d}')

        for batch_idx, batch in enumerate(pbar):
            specs = batch['spectrogram'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            if config.USE_AMP:
                with autocast():
                    outputs, expert_stack, _ = self.model(specs, features)

                    base_loss = self.criterion(outputs, labels)

                    expert_losses = []
                    for i in range(expert_stack.shape[1]):
                        expert_losses.append(self.criterion(expert_stack[:, i, :], labels))
                    expert_loss = torch.stack(expert_losses).mean()

                    loss = base_loss + 0.2 * expert_loss
                    loss = loss / config.ACCUMULATION_STEPS

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.CLIP_GRAD_NORM)

                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()

                    self.optimizer.zero_grad()

            losses.append(loss.item() * config.ACCUMULATION_STEPS)
            _, preds = torch.max(outputs, 1)
            corrects += (preds == labels).sum().item()
            total += labels.size(0)

            if len(losses) > 0:
                pbar.set_postfix({
                    'loss': f'{np.mean(losses[-100:]):.4f}',
                    'acc': f'{corrects/total*100:.1f}%'
                })

        return np.mean(losses) if losses else 0, corrects/total*100 if total > 0 else 0

    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                specs = batch['spectrogram'].to(device)
                features = batch['features'].to(device)
                labels = batch['label'].to(device)

                if config.USE_AMP:
                    with autocast():
                        outputs, _, _ = self.model(specs, features)
                else:
                    outputs, _, _ = self.model(specs, features)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = np.mean(all_preds == all_labels) * 100

        target_mask = np.isin(all_labels, config.PRIMARY_TARGETS)
        target_acc = np.mean(all_preds[target_mask] == all_labels[target_mask]) * 100 if target_mask.sum() > 0 else 0

        f1 = f1_score(all_labels, all_preds, average='weighted')

        class_accs = {}
        for i in range(10):
            mask = all_labels == i
            class_accs[i] = np.mean(all_preds[mask] == i) * 100 if mask.sum() > 0 else 0

        return {
            'accuracy': acc,
            'target_accuracy': target_acc,
            'f1': f1,
            'class_accs': class_accs
        }

    def train(self):
        print("\n" + "="*70)
        print("⭐ CORRECTED HYBRID Training")
        print("="*70)
        print(f"  ⭐ Car Horn & Engine: Real-time precise features")
        print(f"  ⚡ Siren & Others: Cached features")
        print(f"  🎯 Focus: Improve difficult classes (1, 5)")
        print("="*70)

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
            print(f"  Val: acc={val_metrics['accuracy']:.1f}%, 🎯={val_metrics['target_accuracy']:.1f}%")
            print(f"  F1: {val_metrics['f1']:.4f}")

            print("\n  Per-class:")
            for i in range(10):
                mark = "🎯" if i in config.PRIMARY_TARGETS else "  "
                difficult = "⭐" if i in config.DIFFICULT_CLASSES else " "
                acc = val_metrics['class_accs'].get(i, 0)
                print(f"    {mark}{difficult} {config.CLASS_NAMES[i]:16s}: {acc:5.1f}%")

            if val_metrics['target_accuracy'] > self.best_target_acc:
                self.best_target_acc = val_metrics['target_accuracy']
                self.best_acc = val_metrics['accuracy']
                self.patience = 0
                self.save(epoch, val_metrics)
                print(f"\n  ✅ Best! Target={self.best_target_acc:.1f}%")
            else:
                self.patience += 1

            if self.patience >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️ Early stopping")
                break

        print(f"\n✅ Complete! Best target: {self.best_target_acc:.1f}%")
        return self.best_target_acc

    def save(self, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
        }, Path(config.SAVE_DIR) / f"{config.MODEL_NAME}_best.pth")


# ========================= Main =========================
def main():
    print("\n" + "="*70)
    print("⭐ Vehicle Classifier v2.4 - CORRECTED HYBRID")
    print("   어려운 클래스(Car Horn, Engine)에 집중!")
    print("="*70)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_dataset = CorrectedHybridDataset(is_training=True, fold=1)
    val_dataset = CorrectedHybridDataset(is_training=False, fold=1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )

    print("\n🤖 Creating model...")
    model = UltimateEnsemble()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total/1e6:.1f}M")
    print(f"  Focus: ⭐ Car Horn & Engine (precise)")

    trainer = UltimateTrainer(model, train_loader, val_loader)
    best = trainer.train()

    print(f"\n🎉 Best target accuracy: {best:.2f}%")

    return model


if __name__ == '__main__':
    model = main()