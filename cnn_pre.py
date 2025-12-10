#
# import numpy as np
# import pandas as pd
# import librosa
# import sys
# import glob
# import os
# import pyaudio
# import sys
# from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtCore import Qt
# from keras.models import load_model
#
# import pyaudio
# import numpy as np
# import librosa
#
#
# class FullScreenImage(QMainWindow):
#     def __init__(self, image_path):
#         super().__init__()
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         layout = QVBoxLayout(self.central_widget)
#
#         pixmap = QPixmap(image_path)
#         label = QLabel(self)
#         label.setPixmap(pixmap)
#         layout.addWidget(label)
#
#         self.showFullScreen()
#
# def predict():
#     # Load class mapping and model
#
#     app = QApplication(sys.argv)
#     label_to_image = {
#         'air_conditioner': 'path_to_air_conditioner_image.jpg',
#         'car_horn': 'path_to_car_horn_image.jpg',
#         'children_playing': 'path_to_children_playing_image.jpg',
#         'dog_bark': 'path_to_dog_bark_image.jpg',
#         'drilling': 'path_to_drilling_image.jpg',
#         'engine_idling': 'path_to_engine_idling_image.jpg',
#         'gun_shot': 'path_to_gun_shot_image.jpg',
#         'jackhammer': 'path_to_jackhammer_image.jpg',
#         'siren': 'path_to_siren_image.jpg',
#         'street_music': 'path_to_street_music_image.jpg'
#     }
#
#     df = pd.read_csv(r'C:\test\cnn_test\class.csv')
#     model = load_model(r'C:\test\cnn_test\models\Urbansound.h5')
#
#     # Configure PyAudio stream
#     CHUNK = 1024
#     FORMAT = pyaudio.paFloat32
#     CHANNELS = 1
#     RATE = 44100
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK)
#
#     while True:
#         # Initialize data buffer
#         dataBuffer = []
#
#         # Collect 3 seconds of audio data
#         print('Collecting audio data...')
#         for i in range(0, int(RATE / CHUNK * 2.97)):
#             data = stream.read(CHUNK)
#             dataBuffer.append(data)
#
#     # Concatenate audio data and reshape to mel-spectrogram
#         data = b''.join(dataBuffer)
#         data = np.frombuffer(data, dtype=np.float32)
#         ps = librosa.feature.melspectrogram(y=data, sr=RATE, n_mels=128, hop_length=512, n_fft=1024)
#         if ps.shape != (128, 128):
#            ps = librosa.util.fix_length(ps, 128, axis=1)
#         dataSet = np.array([ps.reshape((128, 128, 1))])
#
#     # Make prediction and display result
#         predictions = model.predict(dataSet)[0]
#         predictClass = np.argmax(predictions)
#         resultStr = '{0} {1:.2f}%'.format(df.iloc[predictClass, 1], predictions[predictClass] * 100)
#         print(resultStr)
#
#         # Display the image associated with the predicted label
#         label = df.iloc[predictClass, 1]
#         image_path = label_to_image.get(label)
#         if image_path:
#             window = FullScreenImage(image_path)
#             window.show()
#             app.exec_()
#
#     # Clean up PyAudio stream
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#
# if __name__ == '__main__':
#     predict()


import os
import sys
import threading
import queue
import numpy as np
import pandas as pd
import librosa
import pyaudio
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from keras.models import load_model


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # PyInstaller creates a temp folder and stores path in _MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class FullScreenImage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        self.showFullScreen()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)

    def update_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


def update_image():
    global window
    if not q.empty():
        image_path = q.get()
        if not window:
            window = FullScreenImage()
            window.show()
        window.update_image(image_path)


def audio_collecting_thread(label_to_image, df, model, q):
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

        label = df.iloc[predictClass, 1]
        image_path = label_to_image.get(label)
        if image_path:
            q.put(image_path)

    stream.stop_stream()
    stream.close()
    p.terminate()


def predict():
    app = QApplication(sys.argv)

    label_to_image = {
        'air_conditioner': resource_path('img2/print.jpg'),
        'car_horn': resource_path('img2/carQR.png'),
        'children_playing': resource_path('img2/SeoultechQR1.jpg'),
        'dog_bark': resource_path('img2/dogQR1.jpg'),
        'drilling': resource_path('img2/print.jpg'),
        'engine_idling': resource_path('img2/print.jpg'),
        'gun_shot': resource_path('img2/print.jpg'),
        'jackhammer': resource_path('img2/print.jpg'),
        'siren': resource_path('img2/print.jpg'),
        'street_music': resource_path('img2/streetQR.jpg'),
    }

    df = pd.read_csv(resource_path('class.csv'))
    model = load_model(resource_path('models/best1Urbansound.h5'))

    window = FullScreenImage()
    q = queue.Queue()
    audio_thread = threading.Thread(target=audio_collecting_thread, args=(label_to_image, df, model, q))
    audio_thread.start()

    timer = QTimer()
    timer.timeout.connect(lambda: window.update_image(q.get() if not q.empty() else None))
    timer.start(2970)

    sys.exit(app.exec_())


if __name__ == '__main__':
    predict()






