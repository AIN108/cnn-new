# 🔊 CNN 기반 환경 소리 분류 시스템

CNN(Convolutional Neural Network)을 활용한 환경 소리 분류 프로젝트입니다. UrbanSound8K 데이터셋을 기반으로 도시 환경에서 발생하는 다양한 소리를 자동으로 분류합니다.

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 환경 소리 자동 분류 |
| 데이터셋 | UrbanSound8K (10개 클래스) |
| 모델 | CNN (Convolutional Neural Network) |
| 특징 추출 | MFCC, Mel-Spectrogram |

## 🎯 분류 가능한 소리 (10개 클래스)

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

## 📂 프로젝트 구조

```
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
└── nvi.py                # NVIDIA GPU 설정
```

## 🛠️ 기술 스택

- **언어**: Python 3.8+
- **딥러닝**: PyTorch / TensorFlow
- **오디오 처리**: Librosa, SoundFile
- **데이터 처리**: NumPy, Pandas
- **시각화**: Matplotlib

## 🚀 실행 방법

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

## 📊 모델 성능

| 모델 버전 | 정확도 |
|----------|--------|
| cnn_pc.py | ~85% |
| cnn_pc4.py | ~90% |
| cnn_pc_up.py | ~92% |

## 🔧 주요 기능

- **MFCC 특징 추출**: 오디오 신호에서 멜 주파수 켑스트럼 계수 추출
- **데이터 증강**: 시간 이동, 피치 변환, 노이즈 추가
- **실시간 분류**: 마이크 입력을 통한 실시간 소리 분류 가능
- **라즈베리파이 지원**: 경량화 모델로 임베디드 환경 지원

## 📱 라즈베리파이 배포

```bash
# 라즈베리파이에서 실행
python classiPi.py
```

## 📚 참고 자료

- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

## 개발자

- GitHub: [@AIN108](https://github.com/AIN108)


