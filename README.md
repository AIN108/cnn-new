# CNN 기반 환경 소리 분류 시스템

CNN(Convolutional Neural Network)을 활용한 환경 소리 분류 프로젝트입니다. UrbanSound8K 데이터셋을 기반으로 도시 환경에서 발생하는 다양한 소리를 자동으로 분류합니다.

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 환경 소리 자동 분류 |
| 데이터셋 | UrbanSound8K (10개 클래스) |
| 모델 | CNN (Convolutional Neural Network) |
| 특징 추출 | MFCC, Mel-Spectrogram |

## 분류 가능한 소리 (10개 클래스)

- 에어컨 (air_conditioner)
- 자동차 경적 (car_horn)
- 어린이 놀이 (children_playing)
- 개 짖는 소리 (dog_bark)
- 드릴링 (drilling)
- 엔진 공회전 (engine_idling)
- 총성 (gun_shot)
- 착암기 (jackhammer)
- 사이렌 (siren)
- 거리 음악 (street_music)

## 프로젝트 구조

```text
cnn-new/
├── train.py              # 기본 학습 스크립트
├── trainModel.py         # 모델 학습 (상세 버전)
├── 4trian.py             # 학습 실행 스크립트
├── newtrain.py           # 새 학습 방식
├── newtrain2.py          # 학습 방식 v2
├── imtrain.py            # 이미지 기반 학습
│
├── cnn_pc.py             # CNN 모델 (PC 버전)
├── cnn_pc2.py            # CNN 모델 v2
├── cnn_pc3.py            # CNN 모델 v3
├── cnn_pc4.py            # CNN 모델 v4
├── cnn_pc_up.py          # CNN 모델 업그레이드 버전
├── cnn_pre.py            # CNN 전처리
│
├── ciass.py              # 분류기
├── classiPi.py           # 라즈베리파이용 분류기
│
├── prepare_datasets.py   # 데이터셋 준비
├── quick_start.py        # 빠른 시작 가이드
├── setup_guide.py        # 설정 가이드
├── nvi.py                # NVIDIA GPU 설정
└── LICENSE
```

## 기술 스택

- Python 3.8+
- PyTorch / TensorFlow
- Librosa, SoundFile
- NumPy, Pandas
- Matplotlib

## 실행 방법

### 1. 환경 설정

```bash
pip install torch librosa numpy pandas matplotlib soundfile
```

### 2. 데이터셋 준비

```bash
python prepare_datasets.py
```

### 3. 모델 학습

```bash
python train.py
```

### 4. 분류 실행

```bash
python ciass.py --audio your_audio.wav
```

## 성능 수치에 대한 주의

이 저장소의 과거 학습 스크립트 일부는 UrbanSound8K 메타데이터의 공식 `fold` 구분을 사용하지 않고 `train_test_split`으로 데이터를 다시 나누어 평가했습니다. 따라서 과거 README나 개인 기록에 있던 정확도 수치는 UrbanSound8K 공식 평가 프로토콜과 직접 비교 가능한 대표 성능으로 사용하지 않습니다.

UrbanSound8K는 원본 녹음에서 잘린 관련 오디오 조각이 서로 다른 분할에 섞여 들어가는 데이터 누수를 피하기 위해 제공된 10개 fold를 그대로 사용하는 평가를 권장합니다.

- Dataset / evaluation guidance: https://urbansounddataset.weebly.com/urbansound8k.html

향후 대표 성능은 공식 fold 기준 재평가 결과와 함께 정확도, 클래스별 지표, 모델 크기, Raspberry Pi 추론 지연 등을 별도로 기록할 예정입니다.

## 주요 기능

- MFCC 특징 추출
- Mel-Spectrogram 기반 입력 실험
- 데이터 증강 실험
- 실시간 분류 코드
- Raspberry Pi용 분류 코드

## Raspberry Pi 배포

```bash
python classiPi.py
```

현재 저장소에는 Raspberry Pi용 코드가 있으나, 모델 크기·지연시간·메모리 사용량·전력 소비량에 대한 표준화된 벤치마크는 아직 포함되어 있지 않습니다.

## 참고 자료

- UrbanSound8K Dataset: https://urbansounddataset.weebly.com/urbansound8k.html
- Librosa Documentation: https://librosa.org/doc/latest/index.html

## 라이선스

이 저장소에서 AIN108이 작성한 소스 코드는 MIT License로 배포합니다. 자세한 내용은 `LICENSE`를 참고하십시오. 데이터셋과 외부 Python 패키지는 각각의 별도 이용 조건과 라이선스를 따릅니다.

## 개발자

- GitHub: [@AIN108](https://github.com/AIN108)

## 포트폴리오

- Notion 프로젝트: https://app.notion.com/p/2c5f6964be618083944be742aa949584
