import os
import time
import datetime
import warnings
import librosa
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import glob
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.models import Model
from keras.layers import (Conv2D, BatchNormalization, MaxPooling2D, Dropout,
                          Flatten, Dense, Input, Add, GlobalAveragePooling2D, Activation)
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU 및 Mixed Precision 설정
print("\n" + "=" * 70)
print("GPU 설정")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU 감지: {len(gpus)}개")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 메모리 동적 할당 활성화")

        from tensorflow.keras import mixed_precision

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision 활성화")
    except Exception as e:
        print(f"⚠️  설정 경고: {e}")
else:
    print("⚠️  CPU 모드")

print("=" * 70 + "\n")


def save_weights(computed_weights, manual_weights):
    """클래스 가중치를 파일로 저장"""
    timestr = time.strftime('%Y%m%d-%H%M%S')
    directory = "weights"

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, f'computed_weights_{timestr}.pkl'), 'wb') as f:
        pickle.dump(computed_weights, f)

    with open(os.path.join(directory, f'manual_weights_{timestr}.pkl'), 'wb') as f:
        pickle.dump(manual_weights, f)

    print(f"✅ 가중치 저장 완료: {directory}/")


def load_weights():
    """저장된 클래스 가중치 파일을 불러오기"""
    directory = "weights"

    if not os.path.exists(directory):
        raise Exception(f"{directory} 디렉토리가 없습니다.")

    computed_weights_files = glob.glob(os.path.join(directory, 'computed_weights_*.pkl'))
    manual_weights_files = glob.glob(os.path.join(directory, 'manual_weights_*.pkl'))

    if not computed_weights_files or not manual_weights_files:
        raise Exception("저장된 가중치 파일이 없습니다.")

    computed_weights_files.sort()
    manual_weights_files.sort()

    with open(computed_weights_files[-1], 'rb') as f:
        loaded_computed_weights = pickle.load(f)

    with open(manual_weights_files[-1], 'rb') as f:
        loaded_manual_weights = pickle.load(f)

    return loaded_computed_weights, loaded_manual_weights


def adjust_class_weights_interactive(computed_weights, class_counts):
    """클래스 가중치를 대화형으로 조정"""
    print("\n" + "=" * 70)
    print("클래스별 샘플 수 및 자동 계산된 가중치")
    print("=" * 70)
    print(f"{'클래스ID':<10} {'클래스명':<20} {'샘플 수':<12} {'자동 가중치':<15}")
    print("-" * 70)

    class_names = {
        0: "air_conditioner", 1: "car_horn", 2: "children_playing",
        3: "dog_bark", 4: "drilling", 5: "engine_idling",
        6: "gun_shot", 7: "jackhammer", 8: "siren", 9: "street_music"
    }

    for class_id in sorted(computed_weights.keys()):
        count = class_counts.get(class_id, 0)
        weight = computed_weights[class_id]
        name = class_names.get(class_id, "unknown")
        print(f"{class_id:<10} {name:<20} {count:<12} {weight:<15.4f}")

    print("=" * 70)
    print("\n가중치 조정 옵션:")
    print("1. 자동 계산된 가중치 그대로 사용 (권장)")
    print("2. 특정 클래스의 가중치만 수동 조정")
    print("3. 모든 클래스 가중치 초기화 후 수동 설정")

    choice = input("\n선택하세요(1/2/3, 기본값: 1): ").strip() or "1"
    manual_adjustments = {}

    if choice == "1":
        print("\n✅ 자동 계산된 가중치를 사용합니다.")
        return manual_adjustments

    elif choice == "2":
        print("\n조정할 클래스ID를 입력하세요(쉼표로 구분, 예: 0,3,6)")
        print("입력 없이 Enter를 누르면 자동 가중치를 사용합니다.")
        class_input = input("클래스ID: ").strip()

        if not class_input:
            return manual_adjustments

        try:
            classes_to_adjust = [int(x.strip()) for x in class_input.split(',')]
        except ValueError:
            print("❌ 잘못된 입력입니다. 자동 가중치를 사용합니다.")
            return manual_adjustments

        for class_id in classes_to_adjust:
            if class_id not in computed_weights:
                print(f"⚠️  클래스{class_id}는 존재하지 않습니다.")
                continue

            current_weight = computed_weights[class_id]
            print(f"\n클래스 {class_id} ({class_names.get(class_id, 'unknown')})")
            print(f"  현재 가중치: {current_weight:.4f}")
            print(f"  샘플 수: {class_counts.get(class_id, 0)}")

            try:
                multiplier = float(input(f"  가중치 배수 (예: 1.5, 2.0): ").strip() or "1.0")
                manual_adjustments[class_id] = multiplier
                print(f"  → 새 가중치: {current_weight * multiplier:.4f}")
            except ValueError:
                print(f"  ⚠️  잘못된 입력. 원래 가중치 유지.")

    elif choice == "3":
        print("\n모든 클래스의 가중치를 수동으로 설정합니다.")
        try:
            base_count = float(input("기준 샘플 수 (기본값: 최대 샘플 수): ").strip() or max(class_counts.values()))

            for class_id in sorted(computed_weights.keys()):
                count = class_counts.get(class_id, 1)
                auto_weight = base_count / count
                print(f"\n클래스 {class_id} ({class_names.get(class_id, 'unknown')})")
                print(f"  샘플 수: {count}, 권장 가중치: {auto_weight:.4f}")

                user_input = input(f"  가중치 입력(Enter: 권장값): ").strip()
                if user_input:
                    try:
                        new_weight = float(user_input)
                        manual_adjustments[class_id] = new_weight / computed_weights[class_id]
                    except ValueError:
                        print("  ⚠️  잘못된 입력. 자동값 사용.")
                else:
                    manual_adjustments[class_id] = auto_weight / computed_weights[class_id]
        except ValueError:
            print("❌ 잘못된 입력입니다. 자동 가중치를 사용합니다.")

    return manual_adjustments


def spec_augment(mel_spectrogram, time_mask_param=20, freq_mask_param=20, num_masks=2):
    """SpecAugment 데이터 증강"""
    mel_spec = mel_spectrogram.copy()
    num_mel_channels, num_frames = mel_spec.shape

    for _ in range(num_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, num_frames - t)
        mel_spec[:, t0:t0 + t] = 0

    for _ in range(num_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, num_mel_channels - f)
        mel_spec[f0:f0 + f, :] = 0

    return mel_spec


def importData(apply_augmentation=True):
    """데이터 로딩 및 증강"""
    data = pd.read_csv(r'C:\test\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv')
    valid_data = data[['slice_file_name', 'fold', 'classID', 'classname']][data['end'] - data['start'] >= 0.0]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')

    print(f'📊 데이터 개수: {len(valid_data)}')

    class_frequencies = {i: 0 for i in range(10)}
    D = []
    totalCount = 0

    print('📂 데이터 로딩 시작...')

    for row in valid_data.itertuples():
        if totalCount % 100 == 0:
            print(f'   진행: {totalCount}/{len(valid_data)}', end='\r')

        y, sr = librosa.load(os.path.join(r'C:\test\UrbanSound8K\audio', row.path), duration=2.97, sr=22050)
        class_frequencies[row.classID] += 1

        ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000)
        ps = librosa.power_to_db(ps, ref=np.max)

        if ps.shape[1] < 128:
            ps = np.pad(ps, ((0, 0), (0, 128 - ps.shape[1])))
        elif ps.shape[1] > 128:
            ps = ps[:, :128]

        D.append((ps, row.classID))

        if apply_augmentation:
            for _ in range(2):
                augmented_ps = spec_augment(ps, time_mask_param=15, freq_mask_param=15, num_masks=2)
                D.append((augmented_ps, row.classID))

        totalCount += 1

    print(f'\n✅ 로딩 완료: 원본 {totalCount}개 → 증강 후 {len(D)}개')
    return D, class_frequencies


def compute_class_weights(y):
    """클래스 가중치 자동 계산"""
    y_integers = np.argmax(y, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    return dict(enumerate(class_weights))


class CustomEarlyStopping(Callback):
    """조기 종료 콜백"""

    def __init__(self, threshold=0.97, patience=15, verbose=1, restore_best_weights=True):
        super(CustomEarlyStopping, self).__init__()
        self.stopped_epoch = 0
        self.threshold = threshold
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best = -np.Inf
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_accuracy')
        if current is None:
            return

        if current >= self.threshold:
            if current > self.best:
                self.best = current
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            if self.best >= self.threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.restore_best_weights and self.verbose > 0:
                        print(f'\n✅ 최고 성능 모델 복원 (정확도: {self.best:.4f})')
                        self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch + 1}: 조기 종료')


def residual_block(x, filters, kernel_size=(3, 3), dropout_rate=0.3):
    """Residual Block"""
    fx = Conv2D(filters, kernel_size, padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Activation('relu')(fx)
    fx = Dropout(dropout_rate)(fx)

    fx = Conv2D(filters, kernel_size, padding='same')(fx)
    fx = BatchNormalization()(fx)

    if x.shape[-1] != filters:
        x = Conv2D(filters, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)

    out = Add()([x, fx])
    out = Activation('relu')(out)
    return out


def build_model(input_shape=(128, 128, 1), dropout_rate=0.3, learning_rate=0.001):
    """최적화된 모델 빌드"""
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)

    x = residual_block(x, 64, dropout_rate=dropout_rate)
    x = MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128, dropout_rate=dropout_rate)
    x = MaxPooling2D((2, 2))(x)

    x = residual_block(x, 256, dropout_rate=dropout_rate)

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(10, activation='softmax', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # 1. 데이터 로딩
    print("\n" + "=" * 70)
    print("1단계: 데이터 로딩")
    print("=" * 70)
    dataSet, class_frequencies = importData(apply_augmentation=True)

    X, y = zip(*dataSet)
    X = np.array([x.reshape((128, 128, 1)) for x in X])
    y = np.array(to_categorical(y, 10))

    # 2. 정규화
    print("\n" + "=" * 70)
    print("2단계: 데이터 정규화")
    print("=" * 70)
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_flat)
    X = X_normalized.reshape(X.shape[0], 128, 128, 1)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ 정규화 완료 (scaler.pkl 저장)")

    # 3. 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✅ 데이터 분할: 학습 {len(X_train)}개, 테스트 {len(X_test)}개")

    # 4. 클래스 가중치 설정
    print("\n" + "=" * 70)
    print("3단계: 클래스 가중치 설정")
    print("=" * 70)

    computed_weights = compute_class_weights(y_train)

    # 저장된 가중치 로드 시도
    try:
        loaded_computed_weights, loaded_manual_adjustments = load_weights()
        print("✅ 저장된 가중치를 발견했습니다.")
        use_saved = input("저장된 가중치를 사용하시겠습니까? (예/아니오, 기본값: 예): ").strip().lower()

        if use_saved == "" or use_saved == "예":
            computed_weights = loaded_computed_weights
            manual_adjustments = loaded_manual_adjustments
            print("✅ 저장된 가중치를 사용합니다.")
        else:
            manual_adjustments = adjust_class_weights_interactive(computed_weights, class_frequencies)
    except Exception as e:
        print(f"ℹ️  저장된 가중치 없음: {e}")
        print("새로운 가중치를 설정합니다.\n")
        manual_adjustments = adjust_class_weights_interactive(computed_weights, class_frequencies)

    # 수동 조정 적용
    for class_id, multiplier in manual_adjustments.items():
        computed_weights[class_id] *= multiplier

    print("\n📊 최종 클래스 가중치:")
    for class_id, weight in sorted(computed_weights.items()):
        print(f"  클래스 {class_id}: {weight:.4f}")

    # 가중치 저장
    save_weights(computed_weights, manual_adjustments)

    # 5. Callbacks 설정
    print("\n" + "=" * 70)
    print("4단계: 학습 준비")
    print("=" * 70)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

    checkpoint_filepath = 'best_model_complete.h5'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True,
                                       monitor='val_accuracy', mode='max', verbose=1)

    custom_early_stopping = CustomEarlyStopping(patience=15, threshold=0.96,
                                                restore_best_weights=True, verbose=1)

    # 6. 모델 생성
    print("\n🏗️  모델 생성 중...")
    print("   하이퍼파라미터:")
    print("   - Dropout Rate: 0.3")
    print("   - Learning Rate: 0.001")

    model = build_model(
        input_shape=(128, 128, 1),
        dropout_rate=0.3,
        learning_rate=0.001
    )

    print("\n📋 모델 구조:")
    model.summary()

    # 7. 학습 시작
    print("\n" + "=" * 70)
    print("5단계: 모델 학습")
    print("=" * 70)

    start_time = time.time()

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard, custom_early_stopping, reduce_lr, model_checkpoint],
        class_weight=computed_weights,  # ← 가중치 적용!
        verbose=1
    )

    elapsed_time = time.time() - start_time

    # 8. 평가
    print("\n" + "=" * 70)
    print("6단계: 모델 평가")
    print("=" * 70)
    score = model.evaluate(x=X_test, y=y_test, verbose=1)

    print('\n' + "=" * 70)
    print('✅ 학습 완료!')
    print("=" * 70)
    print(f'⏱️  총 학습 시간: {elapsed_time / 3600:.2f}시간 ({elapsed_time / 60:.1f}분)')
    print(f'📉 Test Loss: {score[0]:.4f}')
    print(f'🎯 Test Accuracy: {score[1]:.4f} ({score[1] * 100:.2f}%)')
    print("=" * 70)

    # 9. 모델 저장
    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = f'complete-sound-classification-{timestr}.h5'
    model_directory = 'models'

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model.save(os.path.join(model_directory, modelName))
    print(f'\n💾 모델 저장: {os.path.join(model_directory, modelName)}')
    print('✨ 모든 작업 완료!')
