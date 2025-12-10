"""
SONYC-UST annotations.csv 구조 분석 스크립트 (수정 버전)
"""

import pandas as pd
from pathlib import Path
from collections import Counter

def analyze_sonyc_annotations():
    """SONYC annotations.csv 구조 상세 분석"""

    sonyc_path = Path(r'C:\cnn\cnn_test\SONYC')
    annotation_file = sonyc_path / 'annotations.csv'

    if not annotation_file.exists():
        print(f"❌ annotations.csv를 찾을 수 없습니다: {annotation_file}")
        return

    print("="*70)
    print("📊 SONYC-UST annotations.csv 분석 (수정 버전)")
    print("="*70)

    df = pd.read_csv(annotation_file)

    print(f"\n📋 기본 정보:")
    print(f"   총 레코드 수: {len(df)}개")
    print(f"   총 컬럼 수: {len(df.columns)}개")

    # split 분포
    if 'split' in df.columns:
        split_counts = df['split'].value_counts()
        print(f"\n📊 Split 분포:")
        for split, count in split_counts.items():
            print(f"   {split:10s}: {count:6d}개 ({count/len(df)*100:.1f}%)")

    # _presence로 끝나는 라벨 컬럼 찾기 (SONYC는 -1, 0, 1 사용)
    print(f"\n🏷️ 라벨 컬럼 분석 (_presence 컬럼):")

    presence_columns = [col for col in df.columns if col.endswith('_presence')]

    print(f"\n✅ 총 {len(presence_columns)}개 _presence 컬럼 발견\n")

    # 라벨별 통계
    label_stats = []
    for col in presence_columns:
        count_positive = (df[col] == 1).sum()
        count_zero = (df[col] == 0).sum()
        count_missing = (df[col] == -1).sum()
        percentage = (count_positive / len(df) * 100)

        label_stats.append({
            'label': col,
            'positive': count_positive,
            'percentage': percentage
        })

        print(f"{col:45s}: +1={count_positive:5d} ({percentage:5.2f}%)  0={count_zero:5d}  -1={count_missing:5d}")

    # UrbanSound8K 클래스와 매핑 가능한 라벨 찾기
    print(f"\n🎯 UrbanSound8K 클래스와 매핑 가능한 SONYC 라벨:")

    urbansound_mapping = {
        'car_horn': ['5-1_car-horn_presence'],
        'engine_idling': ['1-1_small-sounding-engine_presence',
                          '1-2_medium-sounding-engine_presence',
                          '1-3_large-sounding-engine_presence',
                          '1-X_engine-of-uncertain-size_presence',
                          '1_engine_presence'],
        'siren': ['5-3_siren_presence'],
        'dog_bark': ['8-1_dog-barking-whining_presence', '8_dog_presence'],
    }

    print("\n💡 추천 매핑:")
    print("="*70)

    for urbansound_class, sonyc_labels in urbansound_mapping.items():
        print(f"\n{urbansound_class}:")
        available = [label for label in sonyc_labels if label in presence_columns]
        if available:
            for label in available:
                count = (df[label] == 1).sum()
                print(f"   ✓ {label:45s}: {count:5d}개")
        else:
            print(f"   ✗ 매핑 가능한 라벨 없음")

    print("\n" + "="*70)

    # 가장 많이 등장하는 라벨 Top 10
    print(f"\n🔝 가장 많이 등장하는 라벨 Top 10:")
    sorted_labels = sorted(label_stats, key=lambda x: x['positive'], reverse=True)

    for i, stat in enumerate(sorted_labels[:10], 1):
        print(f"   {i:2d}. {stat['label']:45s}: {stat['positive']:5d}개 ({stat['percentage']:5.2f}%)")

    # 실제 오디오 파일 확인
    print(f"\n📂 오디오 파일 확인:")
    audio_files = list(sonyc_path.glob('**/*.wav'))
    print(f"   총 WAV 파일: {len(audio_files)}개")

    if audio_files:
        print(f"\n   샘플 파일 경로:")
        for audio_file in audio_files[:5]:
            print(f"      {audio_file.relative_to(sonyc_path)}")

    # CSV와 오디오 파일 매칭 확인
    if 'audio_filename' in df.columns and audio_files:
        audio_file_dict = {f.name: str(f) for f in audio_files}

        matched = 0
        unmatched_samples = []
        for filename in df['audio_filename'].head(100):
            if filename in audio_file_dict:
                matched += 1
            else:
                if len(unmatched_samples) < 5:
                    unmatched_samples.append(filename)

        print(f"\n✅ 매칭 테스트 (첫 100개):")
        print(f"   CSV에 있는 파일 중 실제 존재: {matched}/100개")

        if unmatched_samples:
            print(f"\n   매칭 안 된 파일 샘플:")
            for f in unmatched_samples:
                print(f"      {f}")

    # 최종 권장사항
    print(f"\n" + "="*70)
    print("📝 최종 권장사항:")
    print("="*70)
    print("""
1. SONYC-UST는 UrbanSound8K의 일부 클래스만 커버함
   - car_horn, engine_idling, siren, dog_bark 등

2. 라벨 값이 -1, 0, 1 세 가지임:
   - -1: 해당 어노테이터가 라벨링 안 함 (제외 필요)
   - 0: 소리 없음
   - 1: 소리 있음

3. 데이터 전처리 시:
   - -1인 샘플은 제외하거나
   - annotator별로 집계해서 다수결로 최종 라벨 결정

4. UrbanSound8K 10개 클래스 중 일부만 학습 가능
   - 나머지 클래스는 다른 데이터셋 필요
    """)


if __name__ == '__main__':
    analyze_sonyc_annotations()