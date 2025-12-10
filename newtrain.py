import os
import time
import datetime
import warnings
import librosa
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import sklearn
import glob
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping
from keras.regularizers import l1_l2, l2
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Activation, \
    GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_tuner import HyperModel, BayesianOptimization
import keras.backend as K

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=5 * 1024)])


# Focal Loss 구현 - 클래스 불균형에 강력함
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)

    return focal_loss_fixed


# 데이터 증강 함수들
def time_shift(audio, shift_max=0.2):
    """시간축 이동"""
    shift = np.random.randint(int(len(audio) * -shift_max), int(len(audio) * shift_max))
    return np.roll(audio, shift)


def pitch_shift(audio, sr, n_steps=2):
    """피치 변환"""
    n_steps = np.random.randint(-n_steps, n_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_noise(audio, noise_factor=0.005):
    """노이즈 추가"""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def time_stretch(audio, rate_range=(0.8, 1.2)):
    """시간 축 늘이기/줄이기"""
    rate = np.random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(audio, rate=rate)


def augment_audio(audio, sr):
    """랜덤하게 증강 적용"""
    augmentations = [
        lambda x: time_shift(x),
        lambda x: pitch_shift(x, sr),
        lambda x: add_noise(x),
        lambda x: time_stretch(x)
    ]

    # 50% 확률로 1-2개의 증강 적용
    if np.random.random() > 0.5:
        num_augs = np.random.randint(1, 3)
        selected_augs = np.random.choice(augmentations, num_augs, replace=False)
        for aug in selected_augs:
            audio = aug(audio)

    return audio


def extract_features(y, sr):
    """향상된 특징 추출"""
    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # MFCC 추가
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # 두 특징을 합침 (128 + 40 = 168 features)
    # 여기서는 mel-spectrogram만 사용하지만, 필요시 MFCC도 추가 가능
    return mel_spec_db


def save_weights(computed_weights, manual_weights):
    timestr = time.strftime('%Y%m%d-%H%M%S')
    directory = "weights"

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, f'computed_weights_{timestr}.pkl'), 'wb') as f:
        pickle.dump(computed_weights, f)

    with open(os.path.join(directory, f'manual_weights_{timestr}.pkl'), 'wb') as f:
        pickle.dump(manual_weights, f)
    print(f"Weights saved in {directory} with timestamp: {timestr}")


def load_weights():
    directory = "weights"
    if not os.path.exists(directory):
        raise Exception(f"{directory} directory does not exist.")

    computed_weights_files = glob.glob(os.path.join(directory, 'computed_weights_*.pkl'))
    manual_weights_files = glob.glob(os.path.join(directory, 'manual_weights_*.pkl'))

    if not computed_weights_files or not manual_weights_files:
        raise Exception("No weights files found.")

    computed_weights_files.sort()
    manual_weights_files.sort()

    with open(computed_weights_files[-1], 'rb') as f:
        loaded_computed_weights = pickle.load(f)

    with open(manual_weights_files[-1], 'rb') as f:
        loaded_manual_weights = pickle.load(f)

    return loaded_computed_weights, loaded_manual_weights


totalRecordCount = 0


def importData(augment=False, augment_factor=2):
    """
    데이터 임포트 및 증강
    augment: 데이터 증강 여부
    augment_factor: 소수 클래스에 대한 증강 배수
    """
    data = pd.read_csv(r'C:\test\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv')

    valid_data = data[['slice_file_name', 'fold', 'classID', 'classname']][data['end'] - data['start'] >= 0.0]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
    print('Import data count:{}'.format(len(valid_data)))

    class_frequencies = {i: 0 for i in range(10)}
    D = []
    totalCount = 0
    progressThreashold = 100

    print('===========Import data begin===========')

    # 첫 번째 패스: 원본 데이터 로드
    original_data = []
    for row in valid_data.itertuples():
        if totalCount % progressThreashold == 0:
            print('Importing data count:{}'.format(totalCount))

        y, sr = librosa.load(os.path.join(r'C:\test\UrbanSound8K\audio', row.path), duration=2.97)
        class_frequencies[row.classID] += 1
        original_data.append((y, sr, row.classID))
        totalCount += 1

    print('\n원본 클래스별 샘플 수:')
    for class_id, frequency in class_frequencies.items():
        print(f"Class {class_id}: {frequency} samples")

    # 데이터 증강 (옵션)
    if augment:
        print('\n데이터 증강 시작...')
        max_samples = max(class_frequencies.values())

        for class_id, frequency in class_frequencies.items():
            if frequency < max_samples * 0.7:  # 70% 미만인 클래스만 증강
                target_samples = int(max_samples * 0.7)
                augment_needed = target_samples - frequency

                print(f'Class {class_id}: {frequency} -> {target_samples} (증강: {augment_needed})')

                class_samples = [(y, sr) for y, sr, cid in original_data if cid == class_id]

                for _ in range(augment_needed):
                    # 랜덤하게 하나 선택하여 증강
                    y, sr = class_samples[np.random.randint(len(class_samples))]
                    augmented = augment_audio(y.copy(), sr)
                    original_data.append((augmented, sr, class_id))
                    class_frequencies[class_id] += 1

    # 특징 추출
    print('\n특징 추출 중...')
    for idx, (y, sr, class_id) in enumerate(original_data):
        if idx % progressThreashold == 0:
            print(f'특징 추출 진행: {idx}/{len(original_data)}')

        ps = extract_features(y, sr)

        if ps.shape[1] < 128:
            ps = np.pad(ps, ((0, 0), (0, 128 - ps.shape[1])))
        elif ps.shape[1] > 128:
            ps = ps[:, :128]

        D.append((ps, class_id))

    print('\n최종 클래스별 샘플 수:')
    for class_id, frequency in class_frequencies.items():
        print(f"Class {class_id}: {frequency} samples")

    print('===========Import data finish===========')

    global totalRecordCount
    totalRecordCount = len(D)
    return D, class_frequencies


def compute_class_weights_improved(y, method='balanced_sqrt'):
    """
    개선된 클래스 가중치 계산
    method: 'balanced', 'balanced_sqrt', 'inverse'
    """
    y_integers = np.argmax(y, axis=1)

    if method == 'balanced':
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(y_integers),
                                                          y=y_integers)
    elif method == 'balanced_sqrt':
        # 더 부드러운 가중치 (제곱근 사용)
        class_counts = np.bincount(y_integers)
        total_samples = len(y_integers)
        n_classes = len(class_counts)
        class_weights = total_samples / (n_classes * np.sqrt(class_counts))
        # 정규화
        class_weights = class_weights / class_weights.sum() * n_classes
    elif method == 'inverse':
        class_counts = np.bincount(y_integers)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)

    return dict(enumerate(class_weights))


class SoundHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes=10, use_focal_loss=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss

    def build(self, hp):
        model = Sequential()

        # Hyperparameters
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
        initial_learning_rate = hp.Float('initial_learning_rate', min_value=0.0001, max_value=0.001,
                                         sampling='LOG', default=0.0003)
        l2_reg = hp.Float('l2_regularization', min_value=0.0001, max_value=0.01,
                          sampling='LOG', default=0.001)

        # 더 깊은 아키텍처
        # Block 1
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape,
                         kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Block 2
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Block 3
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Block 4
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())  # Flatten 대신 사용

        # Dense layers
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam(learning_rate=initial_learning_rate)

        if self.use_focal_loss:
            model.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=0.25),
                          metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                          metrics=['accuracy'])

        return model


class ConfusionMatrixCallback(Callback):
    """에포크마다 confusion matrix를 출력하여 어떤 클래스가 문제인지 파악"""

    def __init__(self, X_val, y_val, every_n_epochs=10):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            from sklearn.metrics import confusion_matrix, classification_report

            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(self.y_val, axis=1)

            print(f'\n에포크 {epoch + 1} - Classification Report:')
            print(classification_report(y_true_classes, y_pred_classes,
                                        target_names=[f'Class {i}' for i in range(10)]))


if __name__ == '__main__':
    # 데이터 증강 여부 선택
    use_augmentation = input("데이터 증강을 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
    augment = use_augmentation != 'n'

    # Focal Loss 사용 여부
    use_focal = input("Focal Loss를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
    use_focal_loss = use_focal != 'n'

    # 데이터 로드
    dataSet, class_frequencies = importData(augment=augment, augment_factor=2)
    X, y = zip(*dataSet)
    X = np.array([x.reshape((128, 128, 1)) for x in X])
    y = np.array(to_categorical(y, 10))

    # Stratified split으로 클래스 비율 유지
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        stratify=np.argmax(y, axis=1))

    print(f"\n학습 데이터: {len(X_train)}, 테스트 데이터: {len(X_test)}")

    # 클래스 가중치 계산 방법 선택
    print("\n클래스 가중치 계산 방법:")
    print("1. balanced (기본)")
    print("2. balanced_sqrt (더 부드러움)")
    print("3. inverse")
    weight_method = input("선택 (1/2/3, 기본값: 2): ").strip() or "2"

    method_map = {"1": "balanced", "2": "balanced_sqrt", "3": "inverse"}
    selected_method = method_map.get(weight_method, "balanced_sqrt")

    computed_weights = compute_class_weights_improved(y_train, method=selected_method)

    print("\n계산된 클래스 가중치:")
    for class_id, weight in sorted(computed_weights.items()):
        print(f"클래스 {class_id}: {weight:.4f}")

    # 수동 조정 옵션
    adjust = input("\n가중치를 수동으로 조정하시겠습니까? (y/n, 기본값: n): ").strip().lower()
    if adjust == 'y':
        print("조정할 클래스 ID와 배수를 입력하세요 (예: 3,1.5)")
        print("여러 개는 세미콜론으로 구분 (예: 3,1.5;6,2.0)")
        adjustments_input = input("입력: ").strip()

        if adjustments_input:
            for adj in adjustments_input.split(';'):
                try:
                    class_id, multiplier = adj.split(',')
                    class_id = int(class_id.strip())
                    multiplier = float(multiplier.strip())
                    computed_weights[class_id] *= multiplier
                    print(f"클래스 {class_id} 가중치: {computed_weights[class_id]:.4f}")
                except:
                    print(f"잘못된 입력: {adj}")

    # 정규화 (선택사항)
    normalize = input("\n가중치를 정규화하시겠습니까? (y/n, 기본값: n): ").strip().lower()
    if normalize == 'y':
        weights_array = np.array(list(computed_weights.values()))
        weights_array = weights_array / weights_array.sum() * len(weights_array)
        computed_weights = dict(enumerate(weights_array))
        print("\n정규화된 가중치:")
        for class_id, weight in sorted(computed_weights.items()):
            print(f"클래스 {class_id}: {weight:.4f}")

    # Callbacks
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                  min_lr=0.00001, verbose=1)

    checkpoint_filepath = 'best_model_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.h5'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                       save_best_only=True,
                                       monitor='val_accuracy',
                                       mode='max',
                                       verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20,
                                   restore_best_weights=True, verbose=1)

    confusion_callback = ConfusionMatrixCallback(X_test, y_test, every_n_epochs=10)

    # Hypermodel 튜닝
    input_shape = (128, 128, 1)
    hypermodel = SoundHyperModel(input_shape, use_focal_loss=use_focal_loss)

    project_name = "sound_opt_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    tuner = BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,  # 더 많은 시도 가능
        seed=42,
        project_name=project_name
    )

    print("\n하이퍼파라미터 튜닝 시작...")
    tuner.search(X_train, y_train,
                 epochs=50,  # 튜닝 시 에포크 수
                 batch_size=32,
                 validation_data=(X_test, y_test),
                 callbacks=[early_stopping],
                 class_weight=computed_weights,
                 verbose=1)

    # 최적 하이퍼파라미터로 최종 학습
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n최적 하이퍼파라미터:")
    print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
    print(f"Learning Rate: {best_hps.get('initial_learning_rate')}")
    print(f"L2 Regularization: {best_hps.get('l2_regularization')}")

    best_model = hypermodel.build(best_hps)

    print("\n최종 모델 학습 시작...")
    history = best_model.fit(
        x=X_train,
        y=y_train,
        epochs=200,  # 최종 학습 시 에포크 수
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard, reduce_lr, model_checkpoint, early_stopping, confusion_callback],
        class_weight=computed_weights,
        verbose=1
    )

    # 최종 평가
    print("\n최종 모델 평가:")
    score = best_model.evaluate(x=X_test, y=y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # 클래스별 정확도 분석
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                   'siren', 'street_music']

    print("\n상세 분류 결과:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)

    # 모델 저장
    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'sound-classification-improved-{}.h5'.format(timestr)
    model_directory = 'models'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    best_model.save(os.path.join(model_directory, modelName))
    print(f'\n모델 저장 완료: {modelName}')

    # 가중치 저장
    save_weights(computed_weights, {})
    print('작업 완료!')