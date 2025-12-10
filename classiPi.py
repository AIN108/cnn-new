import glob
import os
import sys
import time
import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D)
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pickle as pkl
#import sounddevice as sd
import pyaudio
from sklearn.preprocessing import StandardScaler

tf.compat.v1.disable_eager_execution()

import pyaudio
import wave
import librosa
import numpy as np
from tensorflow.keras.models import load_model

def extract_features(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
    mel = np.array(librosa.feature.melspectrogram(X, sr=sample_rate).T)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T)
    return mfccs,chroma,mel,contrast,tonnetz


# Load the saved model
model = tf.keras.models.load_model(r'C:\test\cnn_test\my_model.h5')
# Define the labels
labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# Define the sampling rate and duration
sr = 22050
duration = 2.5

# Define the function for real-time classification
def classify_audio():
    while True:
        # Record audio
        recording = sd.rec(int(sr * duration), samplerate=sr, channels=1)
        sd.wait()

        # Extract features from the recording
        mfccs = np.array(librosa.feature.mfcc(y=np.squeeze(recording), sr=sr, n_mfcc=128).T)
        chroma = np.array(librosa.feature.chroma_stft(y=np.squeeze(recording), sr=sr).T)
        mel = np.array(librosa.feature.melspectrogram(y=np.squeeze(recording), sr=sr).T)
        contrast = np.array(librosa.feature.spectral_contrast(y=np.squeeze(recording), sr=sr).T)
        tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(np.squeeze(recording)), sr=sr).T)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        ext_features = np.expand_dims(ext_features, axis=0)

        # Predict the label
        pred = model.predict(ext_features)
        label_idx = np.argmax(pred)
        label = labels[label_idx]

        # Print the predicted label
        print("Predicted label:", label)

# Call the function to start real-time classification
classify_audio()













