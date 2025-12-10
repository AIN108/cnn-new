"""
차량 소리 분류 모델 - SONYC-UST 통합 버전 (개 짖는 소리 제외)

주요 특징:
- UrbanSound8K + ESC-50 + FSD50K + SONYC-UST 통합
- 클래스 균형 조정 (중앙값 기준)
- 멀티라벨 데이터 처리 (SONYC-UST)
- SONYC에서 dog_bark 클래스 제외 (클래스 불균형 방지)
- 멜 스펙트로그램 기반 CNN 분류기
"""

import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
import librosa

from tqdm import tqdm

# 경고 메시지 숨김
warnings.filterwarnings('ignore')
# CUDA 디버깅을 위한 동기화 모드 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# GPU 사용 가능 시 GPU, 아니면 CPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================= 설정 =========================
class Config:
    """
    모델 학습 및 데이터 처리를 위한 전역 설정 클래스

    SONYC-UST 데이터셋이 추가되어 4개의 데이터셋 통합
    """

    # ===== 데이터셋 경로 =====
    URBANSOUND_PATH = r'C:\cnn\cnn_test\UrbanSound8K'  # UrbanSound8K 데이터셋 루트 경로
    ESC50_PATH = r'C:\cnn\cnn_test\ESC-50-master'      # ESC-50 데이터셋 루트 경로
    FSD50K_PATH = r'C:\cnn\cnn_test\FSD50'             # FSD50K 데이터셋 루트 경로
    SONYC_PATH = r'C:\cnn\cnn_test\SONYC'              # SONYC-UST 데이터셋 루트 경로

    # ===== 오디오 신호 처리 설정 =====
    SAMPLE_RATE = 22050        # 샘플링 레이트 (Hz) - 모든 오디오를 이 주파수로 리샘플링
    AUDIO_DURATION = 4.0       # 오디오 길이 (초) - 모든 클립을 이 길이로 고정
    N_MELS = 128               # 멜 필터뱅크 개수 - 주파수 해상도 결정
    N_FFT = 2048               # FFT 윈도우 크기 - 주파수 분석 해상도
    HOP_LENGTH = 512           # 프레임 간 이동 샘플 수 - 시간 해상도 결정
    F_MIN = 20                 # 최소 주파수 (Hz) - 저주파 컷오프
    F_MAX = 8000               # 최대 주파수 (Hz) - 고주파 컷오프 (Nyquist의 절반 이하)

    # ===== 스펙트로그램 이미지 크기 =====
    SPEC_HEIGHT = 224          # 스펙트로그램 높이 (주파수 축)
    SPEC_WIDTH = 224           # 스펙트로그램 너비 (시간 축)

    # ===== 학습 하이퍼파라미터 =====
    BATCH_SIZE = 32                    # 배치 크기
    LEARNING_RATE = 1e-4               # 초기 학습률 (Adam optimizer)
    NUM_EPOCHS = 200                   # 최대 에포크 수
    EARLY_STOPPING_PATIENCE = 20       # 조기 종료 대기 에포크 수
    VALIDATION_FOLDS = [9, 10]         # UrbanSound8K에서 검증용으로 사용할 폴드 번호

    # ===== 모델 저장 설정 =====
    SAVE_DIR = './saved_models'                        # 모델 저장 디렉토리
    MODEL_NAME = 'vehicle_audio_balanced_with_sonyc'   # 저장할 모델 파일명 접두사

    # ===== 데이터 증강 설정 =====
    USE_AUGMENTATION = True    # 데이터 증강 사용 여부
    NOISE_PROB = 0.3           # 노이즈 추가 확률 (30%)
    NOISE_LEVEL = 0.005        # 노이즈 강도 (진폭 대비)

    # ===== 클래스 정의 =====
    # UrbanSound8K의 10개 클래스 (ID -> 이름 매핑)
    CLASS_NAMES = {
        0: "air_conditioner",   # 에어컨
        1: "car_horn",          # 차량 경적
        2: "children_playing",  # 어린이 놀이
        3: "dog_bark",          # 개 짖음 (SONYC에서는 제외)
        4: "drilling",          # 드릴링
        5: "engine_idling",     # 엔진 공회전
        6: "gun_shot",          # 총성
        7: "jackhammer",        # 착암기
        8: "siren",             # 사이렌
        9: "street_music"       # 거리 음악
    }


# Config 인스턴스 생성 및 저장 디렉토리 생성
config = Config()
os.makedirs(config.SAVE_DIR, exist_ok=True)


# ========================= 명확한 매핑만 사용 =========================
class DatasetMapper:
    """
    외부 데이터셋(ESC-50, FSD50K, SONYC-UST)의 클래스를 UrbanSound8K의 10개 클래스로 매핑

    SONYC-UST는 멀티라벨 데이터셋으로, 각 오디오 파일이 여러 라벨을 가질 수 있음
    이 경우 첫 번째로 매칭되는 라벨만 사용하여 단일 라벨로 변환
    """

    @staticmethod
    def get_esc50_mapping():
        """
        ESC-50 데이터셋의 클래스를 UrbanSound8K 클래스로 매핑

        Returns:
            dict: ESC-50 카테고리명 -> UrbanSound8K 클래스 ID 매핑

        매핑 원칙:
        - 차량 관련 소리는 명확히 구분 (car_horn, engine, siren)
        - 유사한 의미의 소리를 동일 클래스로 그룹화
        - 애매한 경우는 제외
        """
        return {
            # 차량 관련 - 명확한 매핑
            'car_horn': 1,        # 차량 경적 -> car_horn
            'engine': 5,          # 엔진 소리 -> engine_idling
            'siren': 8,           # 사이렌 -> siren

            # 동물/사람 소리
            'dog': 3,             # 개 -> dog_bark
            'crying_baby': 2,     # 울음 소리 -> children_playing

            # 기계/도구 소리
            'vacuum_cleaner': 0,  # 청소기 -> air_conditioner (연속 기계음)
            'glass_breaking': 6,  # 유리 깨지는 소리 -> gun_shot (충격음)
            'chainsaw': 7,        # 전기톱 -> jackhammer (공사 소음)

            # 추가 차량 관련
            'helicopter': 5,      # 헬리콥터 -> engine_idling
            'airplane': 5,        # 비행기 -> engine_idling

            # 추가 연속음
            'breathing': 0,       # 호흡 -> air_conditioner
            'snoring': 0,         # 코골이 -> air_conditioner
            'coughing': 0,        # 기침 -> air_conditioner

            # 추가 사람 소리
            'laughing': 2,        # 웃음 -> children_playing
            'clapping': 2,        # 박수 -> children_playing

            # 추가 동물
            'cat': 3,             # 고양이 -> dog_bark
            'rooster': 3,         # 수탉 -> dog_bark

            # 추가 충격음
            'fireworks': 6,       # 불꽃놀이 -> gun_shot

            # 추가 기계음
            'can_opening': 4,     # 캔 따는 소리 -> drilling
            'keyboard_typing': 4, # 키보드 타이핑 -> drilling

            # 추가 알람/벨
            'church_bells': 8,    # 교회 종 -> siren
            'clock_alarm': 8,     # 알람 시계 -> siren
        }

    @staticmethod
    def get_fsd50k_mapping():
        """
        FSD50K 데이터셋의 클래스를 UrbanSound8K 클래스로 매핑

        Returns:
            dict: FSD50K 라벨명 -> UrbanSound8K 클래스 ID 매핑

        특징:
        - FSD50K는 AudioSet 온톨로지 사용 (계층적 라벨)
        - 상세한 하위 카테고리를 상위 클래스로 통합
        - 차량, 엔진, 사이렌 관련 라벨을 광범위하게 수집
        """
        return {
            # ===== 차량 경적/경보 (클래스 1) =====
            'Vehicle horn, car horn, honking': 1,
            'Car alarm': 1,
            'Bicycle bell': 1,
            'Horn': 1,

            # ===== 엔진 소리 (클래스 5) =====
            'Engine': 5,
            'Engine starting': 5,
            'Idling': 5,                           # 공회전
            'Medium engine (mid frequency)': 5,
            'Light engine (high frequency)': 5,
            'Heavy engine (low frequency)': 5,
            'Accelerating, revving, vroom': 5,     # 가속
            'Car': 5,
            'Bus': 5,
            'Truck': 5,
            'Motorcycle': 5,
            'Motor vehicle (road)': 5,
            'Vehicle': 5,
            'Train': 5,
            'Aircraft': 5,
            'Helicopter': 5,
            'Boat, Water vehicle': 5,

            # ===== 사이렌/알람 (클래스 8) =====
            'Siren': 8,
            'Civil defense siren': 8,              # 민방위 사이렌
            'Police car (siren)': 8,
            'Ambulance (siren)': 8,
            'Fire engine, fire truck (siren)': 8,
            'Emergency vehicle': 8,
            'Alarm': 8,
            'Smoke detector, smoke alarm': 8,
            'Fire alarm': 8,
            'Buzzer': 8,
            'Bell': 8,

            # ===== 공조/환기 (클래스 0) =====
            'Air conditioning': 0,
            'HVAC': 0,                             # 냉난방공조
            'Ventilation fan': 0,
            'Fan': 0,
            'White noise': 0,
            'Mechanical fan': 0,
            'Wind': 0,
            'Hiss': 0,

            # ===== 개 짖음 (클래스 3) =====
            'Bark': 3,
            'Bow-wow': 3,
            'Growling': 3,
            'Dog': 3,
            'Whimper (dog)': 3,
            'Howl': 3,
            'Animal': 3,
            'Domestic animals, pets': 3,

            # ===== 어린이/사람 (클래스 2) =====
            'Child speech, kid speaking': 2,
            'Children shouting': 2,
            'Children playing': 2,
            'Baby cry, infant cry': 2,
            'Child singing': 2,
            'Laughter': 2,
            'Baby laughter': 2,
            'Giggle': 2,
            'Screaming': 2,
            'Whoop': 2,

            # ===== 총성/폭발 (클래스 6) =====
            'Gunshot, gunfire': 6,
            'Machine gun': 6,
            'Cap gun': 6,
            'Explosion': 6,
            'Burst, pop': 6,
            'Fireworks': 6,
            'Firecracker': 6,
            'Bang': 6,

            # ===== 드릴/절삭 (클래스 4) =====
            'Drill': 4,
            'Electric drill': 4,
            'Power tool': 4,
            'Sawing': 4,
            'Sanding': 4,
            'Filing (rasp)': 4,
            'Cutting': 4,

            # ===== 착암기/타격 (클래스 7) =====
            'Jackhammer': 7,
            'Pneumatic hammer': 7,                 # 공압 해머
            'Hammer': 7,
            'Hammering': 7,
            'Slam': 7,

            # ===== 음악/악기 (클래스 9) =====
            'Guitar': 9,
            'Acoustic guitar': 9,
            'Electric guitar': 9,
            'Bass guitar': 9,
            'Piano': 9,
            'Keyboard (musical)': 9,
            'Drum': 9,
            'Drum kit': 9,
            'Singing': 9,
            'Music': 9,
            'Musical instrument': 9,
            'Orchestra': 9,
            'Choir': 9,
            'Plucked string instrument': 9,
        }

    @staticmethod
    def get_sonyc_mapping():
        """
        SONYC-UST 데이터셋의 클래스를 UrbanSound8K 클래스로 매핑

        Returns:
            dict: SONYC-UST 컬럼명 -> UrbanSound8K 클래스 ID 매핑

        SONYC-UST 데이터셋 특징:
        - 뉴욕시 도시 소음 모니터링 프로젝트 데이터
        - annotations.csv에서 각 행은 10초 오디오 파일
        - 각 소리 유형별로 '_presence' 컬럼이 있음
        - 값: 1 (존재), 0 (부재), -1 (불확실/레이블 없음)
        - 멀티라벨 가능: 한 파일에 여러 소리가 동시에 존재 가능

        매핑 전략:
        - SONYC의 세분화된 카테고리를 UrbanSound8K의 10개 클래스로 통합
        - 개 짖는 소리(dog_bark) 클래스는 제외 (클래스 불균형 방지)
        - 차량 관련 소리(경적, 엔진, 사이렌)에 집중
        - 건설 장비 소리 매핑
        """
        return {
            # ===== 차량 경적/경보 (클래스 1: car_horn) =====
            '5-1_car-horn_presence': 1,           # 차량 경적
            '5-2_car-alarm_presence': 1,          # 차량 경보 -> car_horn으로 통합

            # ===== 엔진 소리 (클래스 5: engine_idling) =====
            # SONYC는 엔진을 크기별로 세분화 (small/medium/large/uncertain)
            '1-1_small-sounding-engine_presence': 5,   # 작은 엔진 (오토바이, 소형차)
            '1-2_medium-sounding-engine_presence': 5,  # 중간 엔진 (일반 승용차)
            '1-3_large-sounding-engine_presence': 5,   # 큰 엔진 (트럭, 버스)
            '1-X_engine-of-uncertain-size_presence': 5, # 크기 불명확한 엔진
            '1_engine_presence': 5,                     # 엔진 (통합 라벨)

            # ===== 사이렌/경보 (클래스 8: siren) =====
            '5-3_siren_presence': 8,              # 응급차량 사이렌
            '5-4_reverse-beeper_presence': 8,     # 후진 경보음 -> siren으로 통합
            '5_alert-signal_presence': 8,         # 경보 신호 (통합 라벨)

            # ===== 개 짖음 (클래스 3: dog_bark) - 제외됨 =====
            # 주석 처리: SONYC에서 dog_bark 샘플이 과도하게 많아 클래스 불균형 유발
            # '8-1_dog-barking-whining_presence': 3,  # 개 짖음/낑낑거림
            # '8_dog_presence': 3,                    # 개 소리 (통합)

            # ===== 사람 목소리 (클래스 2: children_playing) =====
            # SONYC의 사람 목소리를 children_playing으로 매핑
            '7-1_person-or-small-group-talking_presence': 2,  # 대화
            '7-2_person-or-small-group-shouting_presence': 2, # 고함
            '7_human-voice_presence': 2,                      # 사람 목소리 (통합)

            # ===== 건설 장비 - 충격/타격 장비 =====
            '2-1_rock-drill_presence': 4,         # 암석 드릴 -> drilling
            '2-2_jackhammer_presence': 7,         # 착암기 -> jackhammer
            '2-3_hoe-ram_presence': 7,            # 유압 해머 -> jackhammer
            '2-4_pile-driver_presence': 7,        # 말뚝 박기 -> jackhammer
            '2_machinery-impact_presence': 7,     # 기계 충격 (통합) -> jackhammer

            # ===== 건설 장비 - 톱 =====
            # 톱 종류별로 drilling 또는 jackhammer로 매핑
            '4-1_chainsaw_presence': 7,           # 전기톱 -> jackhammer (고출력)
            '4-2_small-medium-rotating-saw_presence': 4,  # 중소형 회전톱 -> drilling
            '4-3_large-rotating-saw_presence': 4,         # 대형 회전톱 -> drilling
            '4_powered-saw_presence': 4,                  # 전동톱 (통합) -> drilling

            # ===== 음악 (클래스 9: street_music) =====
            # SONYC는 음악을 고정/이동/아이스크림 트럭으로 세분화
            '6-1_stationary-music_presence': 9,   # 고정 음악 (거리 공연자)
            '6-2_mobile-music_presence': 9,       # 이동 음악 (차량에서 나오는 음악)
            '6-3_ice-cream-truck_presence': 9,    # 아이스크림 트럭 음악
            '6_music_presence': 9,                # 음악 (통합)
        }

    @staticmethod
    def load_esc50():
        """
        ESC-50 데이터셋 전체 로드

        Returns:
            list: 딕셔너리 리스트 [{'path': 파일경로, 'label': 클래스ID, 'dataset': 'esc50'}, ...]

        처리 과정:
        1. meta/esc50.csv 파일에서 메타데이터 로드
        2. 매핑 가능한 카테고리만 필터링
        3. 실제 오디오 파일 존재 여부 확인
        """
        esc_path = Path(config.ESC50_PATH)
        meta_file = esc_path / 'meta' / 'esc50.csv'

        # 메타데이터 파일 확인
        if not meta_file.exists():
            print(f"   [경고] ESC-50 메타데이터를 찾을 수 없습니다")
            return []

        try:
            meta_df = pd.read_csv(meta_file)
        except Exception as e:
            print(f"   [경고] ESC-50 로드 실패: {e}")
            return []

        mapping = DatasetMapper.get_esc50_mapping()
        all_data = []

        # 각 행을 순회하며 매핑 가능한 샘플 수집
        for _, row in meta_df.iterrows():
            category = row['category']
            if category in mapping:  # 매핑 가능한 카테고리만
                audio_path = esc_path / 'audio' / row['filename']
                if audio_path.exists():  # 실제 파일 존재 확인
                    all_data.append({
                        'path': str(audio_path),
                        'label': mapping[category],
                        'dataset': 'esc50'
                    })

        print(f"   [완료] ESC-50: {len(all_data)}개 샘플 로드")
        return all_data

    @staticmethod
    def load_fsd50k():
        """
        FSD50K 데이터셋 전체 로드 (dev + eval 세트)

        Returns:
            list: 딕셔너리 리스트 [{'path': 파일경로, 'label': 클래스ID, 'dataset': 'fsd50k'}, ...]

        처리 과정:
        1. Dev set과 Eval set을 각각 로드
        2. 각 샘플은 복수 라벨을 가질 수 있음 (콤마로 구분)
        3. 첫 번째로 매칭되는 라벨 사용 (우선순위 보장)
        4. 실제 .wav 파일 존재 여부 확인
        """
        fsd_path = Path(config.FSD50K_PATH)
        mapping = DatasetMapper.get_fsd50k_mapping()
        all_data = []

        # ===== Dev set 처리 =====
        dev_meta = fsd_path / 'FSD50K.ground_truth' / 'dev.csv'
        dev_audio = fsd_path / 'FSD50K.dev_audio'

        if dev_meta.exists() and dev_audio.exists():
            df = pd.read_csv(dev_meta)
            count = 0
            for _, row in df.iterrows():
                fname = str(row['fname'])
                # 파일명에 .wav 확장자 추가 (없는 경우)
                if not fname.endswith('.wav'):
                    fname = fname + '.wav'

                # 라벨은 콤마로 구분된 문자열
                labels = str(row['labels']).split(',')

                # 첫 번째로 매칭되는 라벨 사용 (중복 방지)
                for label in labels:
                    label = label.strip()  # 공백 제거
                    if label in mapping:
                        audio_path = dev_audio / fname
                        if audio_path.exists():
                            all_data.append({
                                'path': str(audio_path),
                                'label': mapping[label],
                                'dataset': 'fsd50k'
                            })
                            count += 1
                        break  # 첫 매칭 후 종료
            print(f"   [Dev] FSD50K Dev: {count}개")

        # ===== Eval set 처리 =====
        eval_meta = fsd_path / 'FSD50K.ground_truth' / 'eval.csv'
        eval_audio = fsd_path / 'FSD50K.eval_audio'

        if eval_meta.exists() and eval_audio.exists():
            df = pd.read_csv(eval_meta)
            count = 0
            for _, row in df.iterrows():
                fname = str(row['fname'])
                if not fname.endswith('.wav'):
                    fname = fname + '.wav'

                labels = str(row['labels']).split(',')

                for label in labels:
                    label = label.strip()
                    if label in mapping:
                        audio_path = eval_audio / fname
                        if audio_path.exists():
                            all_data.append({
                                'path': str(audio_path),
                                'label': mapping[label],
                                'dataset': 'fsd50k'
                            })
                            count += 1
                        break
            print(f"   [Eval] FSD50K Eval: {count}개")

        if not all_data:
            print(f"   [경고] FSD50K 데이터를 찾을 수 없습니다")
        else:
            print(f"   [완료] FSD50K 총: {len(all_data)}개 샘플 로드")

        return all_data

    @staticmethod
    def load_sonyc():
        """
        SONYC-UST 데이터셋 전체 로드 (개 짖는 소리 제외)

        Returns:
            list: 딕셔너리 리스트 [{'path': 파일경로, 'label': 클래스ID, 'dataset': 'sonyc',
                                    'source_column': 원본 컬럼명}, ...]

        SONYC-UST 구조:
        - annotations.csv: 메타데이터 파일
          - audio_filename: 오디오 파일명 (예: 12345.wav)
          - [소리유형]_presence: 각 소리 유형의 존재 여부
            * 1: 해당 소리 존재
            * 0: 해당 소리 부재
            * -1: 불확실하거나 레이블 없음
        - 오디오 파일들은 여러 하위 폴더에 분산 저장

        처리 과정:
        1. annotations.csv 로드
        2. 오디오 파일들을 재귀적으로 검색하여 딕셔너리 생성
        3. 각 행마다 매핑된 컬럼 확인
        4. 값이 1인 경우만 데이터로 추가 (0과 -1 제외)
        5. 한 파일에 여러 라벨이 있으면 첫 번째만 사용 (멀티라벨 -> 단일라벨)

        개 짖는 소리 제외 이유:
        - SONYC에서 dog_bark 샘플이 과도하게 많음
        - 클래스 불균형을 심화시켜 모델 성능 저하
        - 다른 데이터셋에서 충분한 dog_bark 샘플 확보 가능
        """
        sonyc_path = Path(config.SONYC_PATH)
        annotation_file = sonyc_path / 'annotations.csv'

        # annotations.csv 파일 확인
        if not annotation_file.exists():
            print(f"   [경고] SONYC annotations.csv를 찾을 수 없습니다: {annotation_file}")
            return []

        print(f"   [완료] SONYC annotations.csv 발견")

        # ===== CSV 로드 =====
        try:
            df = pd.read_csv(annotation_file)
            print(f"   [정보] SONYC 총 레코드: {len(df)}개")
        except Exception as e:
            print(f"   [오류] SONYC CSV 로드 실패: {e}")
            return []

        # ===== 오디오 파일 검색 =====
        # SONYC 파일들은 여러 하위 폴더에 분산되어 있을 수 있음
        # glob로 재귀적으로 모든 .wav 파일 찾기
        print(f"   [검색] 오디오 파일 검색 중...")
        audio_files = {}
        for wav_file in sonyc_path.glob('**/*.wav'):
            # 파일명만 키로 사용 (경로는 값으로 저장)
            audio_files[wav_file.name] = str(wav_file)

        print(f"   [완료] 총 {len(audio_files)}개 오디오 파일 발견")

        # ===== 매핑 및 데이터 수집 =====
        mapping = DatasetMapper.get_sonyc_mapping()
        all_data = []
        class_counts = {}      # 클래스별 수집 개수 추적
        missing_count = 0      # 파일 누락 개수
        skip_minus_one = 0     # 라벨 없음(-1) 스킵 개수

        # 각 행(오디오 파일)마다 처리
        for _, row in df.iterrows():
            filename = row['audio_filename']

            # 오디오 파일 존재 확인
            if filename not in audio_files:
                missing_count += 1
                continue

            # 각 소리 유형별 라벨 컬럼 체크
            matched = False
            for column_name, urban_label in mapping.items():
                # 해당 컬럼이 CSV에 존재하는지 확인
                if column_name in df.columns:
                    value = row[column_name]

                    # -1인 경우: 라벨 없음 또는 불확실 -> 스킵
                    if value == -1:
                        continue

                    # 1인 경우만: 해당 소리가 존재함 -> 데이터 추가
                    if value == 1:
                        all_data.append({
                            'path': audio_files[filename],
                            'label': urban_label,
                            'dataset': 'sonyc',
                            'source_column': column_name  # 디버깅용: 원본 컬럼명 저장
                        })

                        # 클래스별 카운트
                        if urban_label not in class_counts:
                            class_counts[urban_label] = 0
                        class_counts[urban_label] += 1

                        matched = True
                        break  # 첫 번째 매칭만 사용 (멀티라벨 -> 단일라벨)

            # 모든 라벨이 -1이거나 0인 경우
            if not matched and filename in audio_files:
                skip_minus_one += 1

        # ===== 통계 출력 =====
        if missing_count > 0:
            print(f"   [경고] 누락된 파일: {missing_count}개")
        if skip_minus_one > 0:
            print(f"   [정보] 라벨 없음(-1) 스킵: {skip_minus_one}개")

        # 클래스별 수집 통계 (dog_bark 제외됨을 명시)
        if class_counts:
            print(f"   [분포] SONYC 클래스별 수집 (dog_bark 제외):")
            for label in sorted(class_counts.keys()):
                count = class_counts[label]
                print(f"      클래스 {label} ({config.CLASS_NAMES[label]:16s}): {count:5d}개")

        print(f"   [완료] SONYC 총: {len(all_data)}개 샘플 로드")

        return all_data


# ========================= 균형 잡힌 데이터셋 =========================
class BalancedAudioDataset(Dataset):
    """
    클래스 균형을 맞춘 오디오 데이터셋 (SONYC-UST 포함)

    특징:
    - 학습 모드: 모든 데이터셋(UrbanSound8K + ESC-50 + FSD50K + SONYC-UST)을 로드하고 클래스 균형 조정
    - 검증 모드: UrbanSound8K의 특정 폴드만 로드
    - 클래스 균형: 중앙값 기준으로 다운샘플링 (부족한 클래스는 그대로 유지)
    - 데이터 증강: 최소한의 노이즈 추가만 사용
    - SONYC-UST: dog_bark 제외, 멀티라벨을 단일라벨로 변환
    """

    def __init__(self, is_training=True):
        """
        Args:
            is_training (bool): True면 학습용, False면 검증용
        """
        self.is_training = is_training
        self.data = []  # 전체 데이터 리스트
        self.val_paths = set()  # 검증용 파일 경로 (중복 방지용)

        print(f"\n{'=' * 70}")
        print(f"[데이터셋] 로딩 - {'학습' if is_training else '검증'} 모드")
        print(f"{'=' * 70}")

        if is_training:
            # 학습: 모든 데이터 로드 및 균형 조정
            self._load_all_data()
            self._balance_classes()
        else:
            # 검증: UrbanSound8K의 특정 폴드만
            self._load_validation_data()

        print(f"\n[완료] 총 {len(self.data)}개 샘플")
        self._print_distribution()

        # 멜 스펙트로그램 변환기 초기화
        # (파라미터는 Config에서 가져옴)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )

    def _load_all_data(self):
        """
        학습용: 모든 데이터셋 로드 (SONYC-UST 포함!)

        순서:
        1. UrbanSound8K 전체 폴드 (1-10)
        2. ESC-50 전체
        3. FSD50K 전체 (dev + eval)
        4. SONYC-UST 전체 (dog_bark 제외)

        검증용 파일 경로는 별도 저장 (나중에 참고용)
        """
        # 1. UrbanSound8K 전체
        metadata_path = Path(config.URBANSOUND_PATH) / 'metadata' / 'UrbanSound8K.csv'

        if not metadata_path.exists():
            raise FileNotFoundError(f"UrbanSound8K not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path)

        # 검증용 파일 경로 저장 (학습에도 포함되지만 나중에 구분 가능)
        val_metadata = metadata[metadata['fold'].isin(config.VALIDATION_FOLDS)]
        for _, row in val_metadata.iterrows():
            audio_path = Path(config.URBANSOUND_PATH) / 'audio' / f"fold{row['fold']}" / row['slice_file_name']
            if audio_path.exists():
                self.val_paths.add(str(audio_path))

        # 전체 데이터 로드 (모든 폴드 포함)
        for _, row in metadata.iterrows():
            audio_path = Path(config.URBANSOUND_PATH) / 'audio' / f"fold{row['fold']}" / row['slice_file_name']
            if audio_path.exists():
                self.data.append({
                    'path': str(audio_path),
                    'label': row['classID'],
                    'dataset': 'urbansound'
                })

        print(f"   [완료] UrbanSound8K: {len(self.data)}개 (전체)")

        # 2. ESC-50 전체
        esc_data = DatasetMapper.load_esc50()
        self.data.extend(esc_data)

        # 3. FSD50K 전체
        fsd_data = DatasetMapper.load_fsd50k()
        self.data.extend(fsd_data)

        # 4. SONYC-UST 전체 (핵심 추가!)
        sonyc_data = DatasetMapper.load_sonyc()
        self.data.extend(sonyc_data)

    def _load_validation_data(self):
        """
        검증용: UrbanSound8K의 특정 폴드만 로드

        Config.VALIDATION_FOLDS에 지정된 폴드만 사용
        (기본값: [9, 10])
        """
        metadata_path = Path(config.URBANSOUND_PATH) / 'metadata' / 'UrbanSound8K.csv'

        if not metadata_path.exists():
            raise FileNotFoundError(f"UrbanSound8K not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        # 검증용 폴드만 필터링
        metadata = metadata[metadata['fold'].isin(config.VALIDATION_FOLDS)]

        for _, row in metadata.iterrows():
            audio_path = Path(config.URBANSOUND_PATH) / 'audio' / f"fold{row['fold']}" / row['slice_file_name']
            if audio_path.exists():
                self.data.append({
                    'path': str(audio_path),
                    'label': row['classID'],
                    'dataset': 'urbansound'
                })

        print(f"   [완료] UrbanSound8K Fold {config.VALIDATION_FOLDS}: {len(self.data)}개")

    def _balance_classes(self):
        """
        클래스 균형 조정 (학습용 데이터만)

        방법:
        1. 각 클래스별 샘플 수 집계
        2. 중앙값을 목표 샘플 수로 설정
        3. 샘플이 많은 클래스: 무작위 다운샘플링
        4. 샘플이 부족한 클래스: 그대로 유지 (증강 없음)

        이유:
        - 중앙값 사용: 극단적인 불균형의 영향 최소화
        - 증강 없음: 인위적인 데이터 생성으로 인한 과적합 방지
        - SONYC 추가로 car_horn, engine_idling, siren 클래스가 크게 증가
        """
        print(f"\n[균형 조정] 클래스 균형 조정 시작...")

        # 클래스별로 샘플 그룹화
        class_samples = {}
        for item in self.data:
            label = item['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(item)

        # 원본 분포 출력
        print(f"\n[원본 분포] 클래스별 샘플 수:")
        counts = []
        for label in range(10):
            count = len(class_samples.get(label, []))
            counts.append(count)
            print(f"   클래스 {label} ({config.CLASS_NAMES[label]:16s}): {count:5d}개")

        # 목표 샘플 수 결정 (중앙값)
        # 중앙값을 사용하는 이유: 평균보다 극단값의 영향을 덜 받음
        target_samples = int(np.median(counts))
        print(f"\n[목표] 샘플 수: {target_samples}개 (중앙값 기준)")

        # 균형 조정 수행
        balanced_data = []
        for label in range(10):
            samples = class_samples.get(label, [])
            original_count = len(samples)

            if original_count == 0:
                # 샘플이 전혀 없는 경우 (발생하지 않아야 함)
                print(f"   [경고] 클래스 {label}: 샘플 없음")
                continue

            if original_count >= target_samples:
                # 다운샘플링: 무작위로 목표 개수만큼 선택
                selected = random.sample(samples, target_samples)
                balanced_data.extend(selected)
                print(f"   [다운샘플링] 클래스 {label}: {original_count} -> {target_samples}개")
            else:
                # 부족한 경우: 있는 그대로 사용
                balanced_data.extend(samples)
                print(f"   [유지] 클래스 {label}: {original_count}개 (부족)")

        # 균형 잡힌 데이터로 교체 및 셔플
        self.data = balanced_data
        random.shuffle(self.data)

        print(f"\n[완료] 균형 조정 완료: {len(self.data)}개 샘플")

    def _print_distribution(self):
        """
        현재 데이터셋의 클래스 분포 및 데이터셋 비율 출력

        출력 정보:
        1. 클래스별 샘플 수 및 비율
        2. 데이터셋별(UrbanSound8K, ESC-50, FSD50K, SONYC-UST) 샘플 수 및 비율
        """
        # 클래스별 분포
        label_counts = Counter([d['label'] for d in self.data])
        print(f"\n[분포] {'학습' if self.is_training else '검증'} 데이터 클래스 분포:")

        for label in range(10):
            count = label_counts.get(label, 0)
            percentage = (count / len(self.data) * 100) if len(self.data) > 0 else 0
            print(f"   클래스 {label} ({config.CLASS_NAMES[label]:16s}): {count:5d}개 ({percentage:5.1f}%)")

        # 데이터셋별 분포
        dataset_counts = Counter([d['dataset'] for d in self.data])
        print(f"\n[분포] 데이터셋별 샘플 수:")
        for dataset, count in dataset_counts.items():
            percentage = (count / len(self.data) * 100)
            print(f"   {dataset:12s}: {count:5d}개 ({percentage:5.1f}%)")

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플 로드 및 전처리

        Args:
            idx (int): 샘플 인덱스

        Returns:
            tuple: (멜 스펙트로그램 텐서, 라벨)
                - 멜 스펙트로그램: (3, 224, 224) 크기의 텐서 (RGB 형식)
                - 라벨: 0-9 범위의 정수

        처리 과정:
        1. 오디오 파일 로드 (torchaudio 또는 librosa)
        2. 리샘플링 (22050Hz로 통일)
        3. 모노 변환
        4. 길이 조정 (4초로 통일, 짧으면 패딩, 길면 자르기)
        5. 데이터 증강 (학습 시만, 노이즈 추가)
        6. 멜 스펙트로그램 변환
        7. dB 스케일 변환
        8. 정규화 (평균 0, 표준편차 1)
        9. 크기 조정 (224x224)
        10. 3채널 변환 (CNN 입력용)
        """
        item = self.data[idx]

        # ===== 1. 오디오 파일 로드 =====
        try:
            # torchaudio 우선 시도
            waveform, sr = torchaudio.load(item['path'])
        except:
            # 실패 시 librosa로 재시도
            try:
                waveform, sr = librosa.load(item['path'], sr=None)
                waveform = torch.from_numpy(waveform).unsqueeze(0).float()
            except:
                # 모두 실패 시 무음 생성 (에러 방지)
                waveform = torch.zeros(1, int(config.SAMPLE_RATE * config.AUDIO_DURATION))
                sr = config.SAMPLE_RATE

        # ===== 2. 리샘플링 =====
        if sr != config.SAMPLE_RATE:
            resampler = T.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # ===== 3. 모노 변환 =====
        # 스테레오인 경우 채널 평균으로 모노 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ===== 4. 길이 조정 =====
        target_length = int(config.SAMPLE_RATE * config.AUDIO_DURATION)
        if waveform.shape[1] > target_length:
            # 길면 자르기
            if self.is_training:
                # 학습: 무작위 위치에서 자르기 (증강 효과)
                start = random.randint(0, waveform.shape[1] - target_length)
            else:
                # 검증: 중앙에서 자르기 (일관성)
                start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        else:
            # 짧으면 패딩 (끝에 0 추가)
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        # ===== 5. 데이터 증강 (학습 시만) =====
        if self.is_training and config.USE_AUGMENTATION:
            if random.random() < config.NOISE_PROB:
                # 가우시안 노이즈 추가 (30% 확률)
                noise = torch.randn_like(waveform) * config.NOISE_LEVEL
                waveform = waveform + noise
                # 클리핑 (-1 ~ 1 범위 유지)
                waveform = torch.clamp(waveform, -1.0, 1.0)

        # ===== 6. 멜 스펙트로그램 변환 =====
        mel_spec = self.mel_transform(waveform)

        # ===== 7. dB 스케일 변환 =====
        # 진폭 -> dB로 변환 (로그 스케일, 사람 청각과 유사)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # ===== 8. 정규화 =====
        # 평균 0, 표준편차 1로 정규화 (딥러닝 학습 안정화)
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)

        # ===== 9. 크기 조정 =====
        # (N_MELS, time_steps) -> (224, 224) 크기로 리사이즈
        if mel_spec_db.shape[-1] != config.SPEC_WIDTH or mel_spec_db.shape[-2] != config.SPEC_HEIGHT:
            mel_spec_db = F.interpolate(
                mel_spec_db.unsqueeze(0),  # 배치 차원 추가
                size=(config.SPEC_HEIGHT, config.SPEC_WIDTH),
                mode='bilinear',  # 이중선형 보간
                align_corners=False
            ).squeeze(0)  # 배치 차원 제거

        # ===== 10. 3채널 변환 =====
        # (1, 224, 224) -> (3, 224, 224) RGB 형식
        # CNN이 ImageNet 사전학습 가중치를 사용하기 위해 필요
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        return mel_spec_db, item['label']


# ========================= 단순 CNN 모델 =========================
class SimpleCNN(nn.Module):
    """
    VGG 스타일의 단순한 CNN 분류 모델

    구조:
    - 4개의 컨볼루션 블록 (각 블록마다 채널 수 2배 증가)
    - 각 블록: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
    - Adaptive Average Pooling (7x7 고정 크기)
    - 3개의 FC 레이어 (1024 -> 512 -> 10 클래스)

    특징:
    - BatchNorm: 학습 안정화 및 속도 향상
    - Dropout: 과적합 방지 (점진적 증가: 0.1 -> 0.2 -> 0.3 -> 0.5)
    - ReLU: 비선형 활성화 (inplace=True로 메모리 절약)
    """

    def __init__(self, num_classes=10):
        """
        Args:
            num_classes (int): 출력 클래스 수 (기본값: 10)
        """
        super().__init__()

        # ===== 특징 추출 레이어 =====
        self.features = nn.Sequential(
            # Block 1: 3 -> 64 채널
            nn.Conv2d(3, 64, 3, padding=1),      # 3x3 컨볼루션, 패딩으로 크기 유지
            nn.BatchNorm2d(64),                   # 배치 정규화
            nn.ReLU(inplace=True),                # ReLU 활성화
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 2x2 맥스풀링 (크기 1/2)
            nn.Dropout2d(0.1),                    # 10% 드롭아웃 (공간적)

            # Block 2: 64 -> 128 채널
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/4
            nn.Dropout2d(0.2),                    # 20% 드롭아웃

            # Block 3: 128 -> 256 채널
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/8
            nn.Dropout2d(0.2),

            # Block 4: 256 -> 512 채널
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                   # 크기 1/16
            nn.Dropout2d(0.3),                    # 30% 드롭아웃
        )

        # ===== 적응형 평균 풀링 =====
        # 입력 크기와 무관하게 7x7 출력 생성
        # (224x224 입력 -> 14x14 특징맵 -> 7x7로 고정)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ===== 분류 레이어 =====
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),         # 25088 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                       # 50% 드롭아웃 (과적합 방지)
            nn.Linear(1024, 512),                  # 1024 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)            # 512 -> 10 클래스
        )

    def forward(self, x):
        """
        순전파

        Args:
            x (Tensor): 입력 텐서 (batch_size, 3, 224, 224)

        Returns:
            Tensor: 로짓 (batch_size, num_classes)
        """
        x = self.features(x)         # 특징 추출: (B, 3, 224, 224) -> (B, 512, 14, 14)
        x = self.avgpool(x)          # 적응형 풀링: (B, 512, 14, 14) -> (B, 512, 7, 7)
        x = torch.flatten(x, 1)      # 평탄화: (B, 512, 7, 7) -> (B, 25088)
        x = self.classifier(x)       # 분류: (B, 25088) -> (B, 10)
        return x


# ========================= Trainer =========================
class Trainer:
    """
    모델 학습 및 검증을 관리하는 클래스

    주요 기능:
    - 학습 루프 실행
    - 검증 수행 및 메트릭 계산
    - 학습률 스케줄링 (ReduceLROnPlateau)
    - 조기 종료 (Early Stopping)
    - 체크포인트 저장 (최고 성능 모델)
    - 학습 이력 추적
    """

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            config: 설정 객체
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # ===== Optimizer 설정 =====
        # Adam: 적응형 학습률, 모멘텀 기반 최적화
        # weight_decay: L2 정규화 (과적합 방지)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-4
        )

        # ===== 학습률 스케줄러 설정 =====
        # ReduceLROnPlateau: 검증 정확도가 개선되지 않으면 학습률 감소
        # factor=0.5: 학습률을 절반으로 감소
        # patience=5: 5 에포크 동안 개선 없으면 감소
        # min_lr=1e-7: 최소 학습률 제한
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',        # 정확도 최대화 (loss 최소화는 'min')
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )

        # ===== 손실 함수 =====
        # CrossEntropyLoss: 다중 클래스 분류용 (Softmax + NLL Loss)
        self.criterion = nn.CrossEntropyLoss()

        # ===== 학습 상태 변수 =====
        self.best_acc = 0.0              # 최고 검증 정확도
        self.best_epoch = 0              # 최고 성능 에포크
        self.patience_counter = 0        # 조기 종료 카운터

        # ===== 학습 이력 =====
        self.history = {
            'train_loss': [],      # 에포크별 학습 손실
            'train_acc': [],       # 에포크별 학습 정확도
            'val_loss': [],        # 에포크별 검증 손실
            'val_acc': [],         # 에포크별 검증 정확도
            'class_accs': []       # 에포크별 클래스별 정확도
        }

    def train_epoch(self):
        """
        1개 에포크 학습 수행

        Returns:
            tuple: (평균 손실, 정확도)

        처리 과정:
        1. 모델을 학습 모드로 전환
        2. 각 배치마다:
           - 순전파
           - 손실 계산
           - 역전파
           - 가중치 업데이트
        3. 에포크 평균 손실 및 정확도 계산
        """
        self.model.train()  # 학습 모드 (Dropout, BatchNorm 활성화)
        total_loss = 0
        correct = 0
        total = 0

        # 진행바 생성
        pbar = tqdm(self.train_loader, desc='Training', ncols=100)

        for inputs, targets in pbar:
            # 데이터를 GPU로 이동
            inputs, targets = inputs.to(device), targets.to(device)

            # ===== 순전파 및 손실 계산 =====
            self.optimizer.zero_grad()       # 그래디언트 초기화
            outputs = self.model(inputs)     # 예측
            loss = self.criterion(outputs, targets)  # 손실 계산

            # ===== 역전파 및 최적화 =====
            loss.backward()                  # 그래디언트 계산
            self.optimizer.step()            # 가중치 업데이트

            # ===== 통계 업데이트 =====
            total_loss += loss.item()
            _, predicted = outputs.max(1)    # 최대 확률 클래스 선택
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 진행바 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        # 에포크 평균 반환
        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        """
        검증 수행 및 메트릭 계산

        Returns:
            tuple: (평균 손실, 정확도, 클래스별 정확도)

        특징:
        - 그래디언트 계산 비활성화 (메모리 절약)
        - 클래스별 정확도 개별 계산
        - 불균형 데이터셋에서 성능 파악 용이
        """
        self.model.eval()  # 평가 모드 (Dropout 비활성화, BatchNorm 고정)
        total_loss = 0
        correct = 0
        total = 0

        # 클래스별 통계 초기화
        class_correct = [0] * 10
        class_total = [0] * 10

        pbar = tqdm(self.val_loader, desc='Validation', ncols=100)

        # 그래디언트 계산 비활성화 (속도 향상, 메모리 절약)
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                # 순전파
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 통계 업데이트
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 클래스별 통계 업데이트
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i] == targets[i]:
                        class_correct[label] += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        # 평균 메트릭 계산
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        # 클래스별 정확도 출력 및 저장
        print(f"\n[클래스별 정확도]")
        class_accs = {}
        for i in range(10):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                class_accs[i] = acc
                print(f"   클래스 {i} ({config.CLASS_NAMES[i]:16s}): {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                class_accs[i] = 0.0

        return val_loss, val_acc, class_accs

    def train(self):
        """
        전체 학습 루프 실행

        Returns:
            float: 최고 검증 정확도

        처리 과정:
        1. NUM_EPOCHS만큼 반복:
           - 1 에포크 학습
           - 검증 수행
           - 학습률 조정
           - 최고 성능 모델 저장
           - 조기 종료 체크
        2. 최종 모델 저장
        3. 학습 통계 출력
        """
        print(f"\n{'=' * 70}")
        print(f"[학습 시작] SONYC-UST 포함 버전 (개 짖는 소리 제외)")
        print(f"{'=' * 70}\n")

        start_time = time.time()

        for epoch in range(config.NUM_EPOCHS):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
            print(f"{'=' * 50}")

            # ===== 학습 및 검증 =====
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, class_accs = self.validate()

            # ===== 학습률 스케줄링 =====
            # 검증 정확도 기반으로 학습률 조정
            self.scheduler.step(val_acc)

            # ===== 이력 저장 =====
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['class_accs'].append(class_accs)

            # ===== 결과 출력 =====
            print(f"\n[결과]")
            print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # ===== 최고 성능 체크 및 저장 =====
            if val_acc > self.best_acc:
                improvement = val_acc - self.best_acc
                self.best_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0  # 카운터 리셋
                self.save_checkpoint(epoch)
                print(f"   [최고 성능] 갱신! (+{improvement:.2f}%)")
            else:
                # 개선 없음
                self.patience_counter += 1
                print(f"   [대기] No improvement: {self.patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            # ===== 조기 종료 체크 =====
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n[조기 종료] Early Stopping at epoch {epoch + 1}")
                break

        # ===== 학습 완료 =====
        total_time = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"[학습 완료]")
        print(f"   총 시간: {str(timedelta(seconds=int(total_time)))}")
        print(f"   최고 정확도: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"{'=' * 70}\n")

        self.save_final()
        return self.best_acc

    def save_checkpoint(self, epoch):
        """
        최고 성능 체크포인트 저장

        Args:
            epoch (int): 현재 에포크

        저장 내용:
        - 모델 가중치
        - Optimizer 상태 (학습 재개 가능)
        - 최고 정확도
        - 학습 이력
        - 설정 정보
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
            'config': vars(config)  # Config 객체를 딕셔너리로 변환
        }
        path = os.path.join(config.SAVE_DIR, f'{config.MODEL_NAME}_best.pth')
        torch.save(checkpoint, path)

    def save_final(self):
        """
        최종 모델 저장 (추론용, 가벼운 버전)

        저장 내용:
        - 모델 가중치만 (Optimizer 제외)
        - 최고 정확도
        - 학습 이력
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'config': vars(config)
        }
        path = os.path.join(config.SAVE_DIR, f'{config.MODEL_NAME}_final.pth')
        torch.save(checkpoint, path)
        print(f"[모델 저장] {path}")


# ========================= 분류기 =========================
class AudioClassifier:
    """
    학습된 모델을 사용한 오디오 분류기

    기능:
    - 저장된 모델 로드
    - 단일 오디오 파일 분류
    - 배치 오디오 파일 분류
    - 확률 분포 반환
    """

    def __init__(self, model_path):
        """
        Args:
            model_path (str): 저장된 모델 파일 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ===== 모델 로드 =====
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SimpleCNN(num_classes=10)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # 평가 모드로 설정

        print(f"[모델 로드] {model_path}")
        print(f"   최고 정확도: {checkpoint['best_acc']:.2f}%")

        # ===== 멜 스펙트로그램 변환기 초기화 =====
        # 학습 시와 동일한 파라미터 사용
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )

    def preprocess_audio(self, audio_path):
        """
        오디오 파일 전처리 (학습 데이터와 동일한 방식)

        Args:
            audio_path (str): 오디오 파일 경로

        Returns:
            Tensor: 전처리된 텐서 (1, 3, 224, 224) - 배치 차원 포함

        처리 과정:
        1. 오디오 로드
        2. 리샘플링
        3. 모노 변환
        4. 길이 조정 (중앙에서 자르기, 증강 없음)
        5. 멜 스펙트로그램 변환
        6. dB 변환 및 정규화
        7. 크기 조정
        8. 3채널 변환
        """
        # ===== 1. 오디오 로드 =====
        try:
            waveform, sr = torchaudio.load(audio_path)
        except:
            waveform, sr = librosa.load(audio_path, sr=None)
            waveform = torch.from_numpy(waveform).unsqueeze(0).float()

        # ===== 2. 리샘플링 =====
        if sr != config.SAMPLE_RATE:
            resampler = T.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # ===== 3. 모노 변환 =====
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ===== 4. 길이 조정 =====
        target_length = int(config.SAMPLE_RATE * config.AUDIO_DURATION)
        if waveform.shape[1] > target_length:
            # 중앙에서 자르기 (검증과 동일)
            start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        else:
            # 패딩
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))

        # ===== 5-8. 스펙트로그램 변환 및 정규화 =====
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)

        # 크기 조정
        mel_spec_db = F.interpolate(
            mel_spec_db.unsqueeze(0),
            size=(config.SPEC_HEIGHT, config.SPEC_WIDTH),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # 3채널 변환
        mel_spec_db = mel_spec_db.repeat(3, 1, 1)

        # 배치 차원 추가 및 반환
        return mel_spec_db.unsqueeze(0)

    def predict(self, audio_path):
        """
        단일 오디오 파일 분류

        Args:
            audio_path (str): 오디오 파일 경로

        Returns:
            dict: 분류 결과
                - class_id: 예측 클래스 ID (0-9)
                - class_name: 예측 클래스 이름
                - confidence: 예측 확신도 (0-1)
                - probabilities: 전체 클래스 확률 분포
        """
        # 전처리
        input_tensor = self.preprocess_audio(audio_path).to(self.device)

        # 추론 (그래디언트 계산 비활성화)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Softmax로 확률 변환
            probabilities = F.softmax(outputs, dim=1)[0]
            # 최대 확률 클래스 선택
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        return {
            'class_id': predicted_class,
            'class_name': config.CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': {
                config.CLASS_NAMES[i]: prob.item()
                for i, prob in enumerate(probabilities)
            }
        }

    def predict_batch(self, audio_paths):
        """
        여러 오디오 파일 일괄 분류

        Args:
            audio_paths (list): 오디오 파일 경로 리스트

        Returns:
            list: 각 파일의 분류 결과 리스트
        """
        results = []
        for path in tqdm(audio_paths, desc='Classifying'):
            try:
                result = self.predict(path)
                results.append({'path': path, **result})
            except Exception as e:
                # 에러 발생 시에도 계속 진행
                print(f"[오류] ({path}): {e}")
                results.append({'path': path, 'error': str(e)})
        return results


# ========================= 메인 함수 =========================
def main():
    """
    전체 파이프라인 실행

    순서:
    1. 환경 정보 출력
    2. 데이터셋 로드 및 전처리 (4개 데이터셋 통합)
    3. 데이터로더 생성
    4. 모델 초기화
    5. 학습 수행
    6. 모델 저장
    7. 분류기 초기화 (테스트용)
    """
    # ===== 1. 환경 정보 출력 =====
    print(f"\n[디바이스] {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n{'=' * 70}")
    print(f"차량 소리 분류 - SONYC-UST 통합 버전 (개 짖는 소리 제외)")
    print(f"{'=' * 70}")
    print(f"[데이터셋 경로]")
    print(f"   UrbanSound8K: {config.URBANSOUND_PATH}")
    print(f"   ESC-50: {config.ESC50_PATH}")
    print(f"   FSD50K: {config.FSD50K_PATH}")
    print(f"   SONYC-UST: {config.SONYC_PATH}")
    print(f"[설정]")
    print(f"   저장 경로: {config.SAVE_DIR}")
    print(f"   에포크: {config.NUM_EPOCHS}")
    print(f"   Early Stopping: {config.EARLY_STOPPING_PATIENCE}")
    print(f"   학습률: {config.LEARNING_RATE}")
    print(f"   배치 크기: {config.BATCH_SIZE}")
    print(f"   클래스 균형: 중앙값 기준")
    print(f"[특이사항]")
    print(f"   SONYC dog_bark 제외 (클래스 불균형 방지)")
    print(f"{'=' * 70}\n")

    # ===== 2. 시드 고정 (재현성 확보) =====
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ===== 3. 데이터셋 로드 =====
    train_dataset = BalancedAudioDataset(is_training=True)   # 학습용
    val_dataset = BalancedAudioDataset(is_training=False)    # 검증용

    # ===== 4. 데이터로더 생성 =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,          # 학습 시 셔플
        num_workers=4,         # 멀티프로세싱 (데이터 로딩 속도 향상)
        pin_memory=True        # GPU 전송 속도 향상
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,         # 검증 시 셔플 안 함
        num_workers=4,
        pin_memory=True
    )

    # ===== 5. 모델 초기화 =====
    model = SimpleCNN(num_classes=10)
    print(f"\n[모델] 파라미터: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    # ===== 6. 학습 수행 =====
    trainer = Trainer(model, train_loader, val_loader, config)
    best_acc = trainer.train()

    # ===== 7. 결과 출력 =====
    print(f"\n[최종 결과] 정확도: {best_acc:.2f}%")
    print(f"\n[저장된 모델]")
    print(f"   - Best: {os.path.join(config.SAVE_DIR, config.MODEL_NAME + '_best.pth')}")
    print(f"   - Final: {os.path.join(config.SAVE_DIR, config.MODEL_NAME + '_final.pth')}")

    # ===== 8. 분류기 테스트 =====
    print(f"\n{'=' * 70}")
    print(f"[분류기 테스트]")
    print(f"{'=' * 70}\n")

    model_path = os.path.join(config.SAVE_DIR, f'{config.MODEL_NAME}_best.pth')
    classifier = AudioClassifier(model_path)

    print(f"\n[완료] 분류기 준비 완료!")


# ===== 프로그램 진입점 =====
if __name__ == '__main__':
    main()