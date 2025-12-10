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
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_tuner import HyperModel, BayesianOptimization

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=5 * 1024)])


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
        raise Exception(f"{directory} directory does not exist. Please run the program to compute weights first.")

    computed_weights_files = glob.glob(os.path.join(directory, 'computed_weights_*.pkl'))
    manual_weights_files = glob.glob(os.path.join(directory, 'manual_weights_*.pkl'))

    if not computed_weights_files or not manual_weights_files:
        raise Exception("No weights files found in the weights directory.")

    computed_weights_files.sort()
    manual_weights_files.sort()

    with open(computed_weights_files[-1], 'rb') as f:
        loaded_computed_weights = pickle.load(f)

    with open(manual_weights_files[-1], 'rb') as f:
        loaded_manual_weights = pickle.load(f)

    return loaded_computed_weights, loaded_manual_weights


def adjust_class_weights_interactive(computed_weights, class_counts):
    """
    클래스 가중치를 대화형으로 조정하는 함수
    """
    print("\n" + "=" * 70)
    print("클래스별 샘플 수 및 자동 계산된 가중치")
    print("=" * 70)
    print(f"{'클래스 ID':<10} {'클래스명':<20} {'샘플 수':<12} {'자동 가중치':<15}")
    print("-" * 70)

    class_names = {
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

    for class_id in sorted(computed_weights.keys()):
        count = class_counts.get(class_id, 0)
        weight = computed_weights[class_id]
        name = class_names.get(class_id, "unknown")
        print(f"{class_id:<10} {name:<20} {count:<12} {weight:<15.4f}")

    print("=" * 70)

    # 가중치 조정 방법 선택
    print("\n가중치 조정 옵션:")
    print("1. 자동 계산된 가중치 그대로 사용")
    print("2. 특정 클래스의 가중치만 수동 조정")
    print("3. 모든 클래스 가중치 초기화 후 수동 설정")

    choice = input("\n선택하세요 (1/2/3, 기본값: 1): ").strip() or "1"

    manual_adjustments = {}

    if choice == "1":
        print("\n자동 계산된 가중치를 사용합니다.")
        return manual_adjustments

    elif choice == "2":
        print("\n조정할 클래스 ID를 입력하세요 (쉼표로 구분, 예: 0,3,6)")
        print("입력 없이 Enter를 누르면 자동 가중치를 사용합니다.")
        class_input = input("클래스 ID: ").strip()

        if not class_input:
            print("자동 계산된 가중치를 사용합니다.")
            return manual_adjustments

        try:
            classes_to_adjust = [int(x.strip()) for x in class_input.split(',')]
        except ValueError:
            print("잘못된 입력입니다. 자동 가중치를 사용합니다.")
            return manual_adjustments

        for class_id in classes_to_adjust:
            if class_id not in computed_weights:
                print(f"클래스 {class_id}는 존재하지 않습니다. 건너뜁니다.")
                continue

            current_weight = computed_weights[class_id]
            print(f"\n클래스 {class_id} ({class_names.get(class_id, 'unknown')})")
            print(f"현재 가중치: {current_weight:.4f}")
            print(f"샘플 수: {class_counts.get(class_id, 0)}")

            adjustment_method = input("조정 방법 (1: 배수로 조정, 2: 직접 값 입력, 기본값: 1): ").strip() or "1"

            try:
                if adjustment_method == "1":
                    multiplier = float(input(f"가중치를 몇 배로 조정하시겠습니까? (예: 1.5, 2.0): ").strip() or "1.0")
                    manual_adjustments[class_id] = multiplier
                    print(f"→ 새 가중치: {current_weight * multiplier:.4f}")
                elif adjustment_method == "2":
                    new_weight = float(input(f"새로운 가중치 값을 입력하세요: ").strip())
                    manual_adjustments[class_id] = new_weight / current_weight
                    print(f"→ 새 가중치: {new_weight:.4f}")
            except ValueError:
                print(f"잘못된 입력입니다. 클래스 {class_id}는 원래 가중치를 유지합니다.")

    elif choice == "3":
        print("\n모든 클래스의 가중치를 수동으로 설정합니다.")
        print("기준 샘플 수를 입력하세요 (예: 1000):")

        try:
            base_count = float(input("기준 샘플 수: ").strip() or max(class_counts.values()))

            for class_id in sorted(computed_weights.keys()):
                count = class_counts.get(class_id, 1)
                auto_weight = base_count / count
                print(f"\n클래스 {class_id} ({class_names.get(class_id, 'unknown')})")
                print(f"샘플 수: {count}, 권장 가중치: {auto_weight:.4f}")

                user_input = input(f"가중치 입력 (Enter: 권장값 사용): ").strip()
                if user_input:
                    try:
                        new_weight = float(user_input)
                        manual_adjustments[class_id] = new_weight / computed_weights[class_id]
                    except ValueError:
                        print("잘못된 입력입니다. 자동 계산 값을 사용합니다.")
                else:
                    manual_adjustments[class_id] = auto_weight / computed_weights[class_id]
        except ValueError:
            print("잘못된 입력입니다. 자동 가중치를 사용합니다.")

    return manual_adjustments


totalRecordCount = 0


def importData():
    data = pd.read_csv(r'C:\test\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv')

    valid_data = data[['slice_file_name', 'fold', 'classID', 'classname']][data['end'] - data['start'] >= 0.0]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
    print('Import data count:{}'.format(len(valid_data)))
    class_frequencies = {i: 0 for i in range(10)}
    D = []
    totalCount = 0
    progressThreashold = 100
    print('===========Import data begin===========')
    for row in valid_data.itertuples():
        if totalCount % progressThreashold == 0:
            print('Importing data count:{}'.format(totalCount))
        y, sr = librosa.load(os.path.join(r'C:\test\UrbanSound8K\audio', row.path), duration=2.97)
        class_frequencies[row.classID] += 1
        ps = librosa.feature.melspectrogram(y=y, sr=sr)

        if ps.shape[1] < 128:
            ps = np.pad(ps, ((0, 0), (0, 128 - ps.shape[1])))
        elif ps.shape[1] > 128:
            ps = ps[:, :128]

        D.append((ps, row.classID))
        totalCount += 1

    print('\n클래스별 샘플 수:')
    for class_id, frequency in class_frequencies.items():
        print(f"Class {class_id}: {frequency} samples")

    print('===========Import data finish===========')

    global totalRecordCount
    totalRecordCount = totalCount
    return D, class_frequencies


def compute_class_weights(y):
    y_integers = np.argmax(y, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    return dict(enumerate(class_weights))


class CustomEarlyStopping(Callback):
    def __init__(self, threshold=0.98, patience=10, verbose=1, restore_best_weights=True):
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
                    if self.restore_best_weights:
                        if self.verbose > 0:
                            print(f'Restoring model weights from the end of the best epoch: {self.stopped_epoch + 1}')
                        self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch + 1}: early stopping')


class SoundHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()

        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1, default=0.5)
        initial_learning_rate = hp.Float('initial_learning_rate', min_value=0.001, max_value=0.01, sampling='LOG',
                                         default=0.003)

        model.add(Conv2D(16, (5, 5), strides=(1, 1), input_shape=self.input_shape, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(10, activation='softmax'))

        optimizer = Adam(learning_rate=initial_learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

        return model


if __name__ == '__main__':
    dataSet, class_frequencies = importData()
    X, y = zip(*dataSet)
    X = np.array([x.reshape((128, 128, 1)) for x in X])
    y = np.array(to_categorical(y, 10))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 클래스 가중치 계산
    computed_weights = compute_class_weights(y_train)

    # 저장된 가중치 로드 시도
    try:
        loaded_computed_weights, loaded_manual_adjustments = load_weights()
        print("저장된 가중치를 불러왔습니다.")
        use_saved = input("저장된 가중치를 사용하시겠습니까? (예/아니오, 기본값: 예): ").strip().lower()

        if use_saved == "" or use_saved == "예":
            computed_weights = loaded_computed_weights
            manual_adjustments = loaded_manual_adjustments
            print("저장된 가중치를 사용합니다.")
        else:
            manual_adjustments = adjust_class_weights_interactive(computed_weights, class_frequencies)
    except Exception as e:
        print(f"저장된 가중치 없음: {e}")
        print("새로운 가중치를 설정합니다.\n")
        manual_adjustments = adjust_class_weights_interactive(computed_weights, class_frequencies)

    # 수동 조정 적용
    for class_id, multiplier in manual_adjustments.items():
        computed_weights[class_id] *= multiplier

    print("\n최종 클래스 가중치:")
    for class_id, weight in sorted(computed_weights.items()):
        print(f"클래스 {class_id}: {weight:.4f}")

    # 가중치 저장
    save_weights(computed_weights, manual_adjustments)

    # Callbacks
    tensorboard = TensorBoard(log_dir="logs/", histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.001, verbose=1)
    checkpoint_filepath = 'best_model.h5'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True,
                                                monitor='val_accuracy', mode='max')

    # Hypermodel 튜닝 및 학습
    input_shape = (128, 128, 1)
    hypermodel = SoundHyperModel(input_shape)

    project_name = "sound_optimization_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    tuner = BayesianOptimization(hypermodel, objective='val_accuracy', max_trials=10, seed=42,
                                 project_name=project_name)

    custom_early_stopping = CustomEarlyStopping(patience=10, threshold=0.95, restore_best_weights=True, verbose=1)

    tuner.search(X_train, y_train, epochs=300, validation_data=(X_test, y_test), callbacks=[custom_early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = hypermodel.build(best_hps)
    best_model.fit(x=X_train, y=y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                   callbacks=[tensorboard, custom_early_stopping, reduce_lr, model_checkpoint_callback],
                   class_weight=computed_weights)

    score = best_model.evaluate(x=X_test, y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'sound-classification-{}.h5'.format(timestr)
    model_directory = 'models'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    best_model.save(os.path.join(model_directory, modelName))
    print('Model exported and finished')


