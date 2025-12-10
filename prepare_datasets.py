
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def prepare_fsd50k_vehicle_classes(fsd_path):
    """FSD50K에서 차량 관련 클래스만 추출"""

    # UrbanSound8K 클래스로 매핑
    fsd_to_urban = {
        'Car_horn': 1,      # car_horn
        'Vehicle_horn': 1,
        'Car_alarm': 1,
        'Engine': 5,        # engine_idling
        'Engine_starting': 5,
        'Motor_vehicle_(road)': 5,
        'Car': 5,
        'Car_passing_by': 5,
        'Truck': 5,
        'Bus': 5,
        'Motorcycle': 5,
        'Siren': 8,         # siren
        'Ambulance_(siren)': 8,
        'Fire_engine,_fire_truck_(siren)': 8,
        'Police_car_(siren)': 8,
        'Civil_defense_siren': 8
    }

    # Ground truth 파일 로드
    dev_gt = pd.read_csv(fsd_path / 'FSD50K.ground_truth/dev.csv')
    eval_gt = pd.read_csv(fsd_path / 'FSD50K.ground_truth/eval.csv')

    # 차량 관련 샘플만 필터링
    vehicle_samples = []

    for df, split in [(dev_gt, 'dev'), (eval_gt, 'eval')]:
        for _, row in df.iterrows():
            labels = row['labels'].split(',')
            for label in labels:
                if label in fsd_to_urban:
                    vehicle_samples.append({
                        'fname': row['fname'],
                        'split': split,
                        'original_label': label,
                        'urban_label': fsd_to_urban[label]
                    })

    print(f"✅ 차량 관련 샘플 {len(vehicle_samples)}개 추출")
    return pd.DataFrame(vehicle_samples)

def resample_audio(input_path, output_path, target_sr=22050):
    """오디오 리샘플링"""
    y, sr = librosa.load(input_path, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, y, target_sr)

print("데이터셋 전처리 함수 준비 완료!")
        