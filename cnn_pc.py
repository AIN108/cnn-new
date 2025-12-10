import os
import sys
import numpy as np
import pandas as pd
import librosa
import pyaudio
from keras.models import load_model
import keras.backend as K


# Focal Loss 정의 (모델 학습 시 사용한 것과 동일해야 함)
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)

    return focal_loss_fixed


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 오디오 수집 부분, 건드리지 마시오
def audio_collecting_thread(target_labels, df, model):
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print('오디오 수집 시작...')
    # 여기까지
    while True:
        dataBuffer = []
        print('Collecting audio data...')
        for i in range(0, int(RATE / CHUNK * 2.97)):
            data = stream.read(CHUNK)
            dataBuffer.append(data)
        data = b''.join(dataBuffer)
        data = np.frombuffer(data, dtype=np.float32)
        ps = librosa.feature.melspectrogram(y=data, sr=RATE, n_mels=128, hop_length=512, n_fft=1024)
        if ps.shape != (128, 128):
            ps = librosa.util.fix_length(ps, size=128, axis=1)
        dataSet = np.array([ps.reshape((128, 128, 1))])
        predictions = model.predict(dataSet)[0]
        predictClass = np.argmax(predictions)
        resultStr = '{0} {1:.2f}%'.format(df.iloc[predictClass, 1], predictions[predictClass] * 100)
        print(resultStr)
        # 타겟 소리 감지 시 추가 메시지 출력
        label = df.iloc[predictClass, 1]
        if label in target_labels:
            print(f'Detected: {label}')
    stream.stop_stream()
    stream.close()
    p.terminate()


# 원하는 라벨에 따라 detect가 나오도록 설정되어 있음
# 0: "air_conditioner"    # 에어컨
# 1: "car_horn"           # 자동차 경적
# 2: "children_playing"   # 아이들 노는 소리
# 3: "dog_bark"           # 개 짖는 소리
# 4: "drilling"           # 드릴 소리
# 5: "engine_idling"      # 엔진 공회전
# 6: "gun_shot"           # 총소리
# 7: "jackhammer"         # 착암기(공사장 해머)
# 8: "siren"              # 사이렌
# 9: "street_music"       # 거리 음악
def predict():
    target_labels = ['car_horn', 'engine_idling', 'siren']
    df = pd.read_csv(resource_path('class.csv'))

    # 모델 로드 시 custom_objects에 focal_loss_fixed 전달
    model = load_model(
        resource_path('models/sound-classification-improved.h5'),
        custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)}
    )

    audio_collecting_thread(target_labels, df, model)


if __name__ == '__main__':
    predict()