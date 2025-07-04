# -*- coding: utf-8 -*-
"""test_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nxxg-iX8-yHvMPD6l9tM7rOq5fGn7KcT
"""

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import sys


class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


model = load_model("model.h5", custom_objects={"Attention": Attention})

class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def extract_log_mel(path, sr=22050, duration=3, n_mels=128):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)), mode='constant')
    else:
        y = y[:sr * duration]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < 128:
        pad_width = 128 - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_db = S_db[:, :128]
    S_db = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)
    return S_db[..., np.newaxis]


def predict_emotion(file_path):
    mel = extract_log_mel(file_path)
    mel = np.expand_dims(mel, axis=0)
    pred = model.predict(mel)
    emotion = class_names[np.argmax(pred)]
    return emotion

file_path = "/content/mixkit-small-group-cheer-and-applause-518.wav"
try:
    result = predict_emotion(file_path)
    print(f" Predicted Emotion: {result}")
except Exception as e:
    print(" Error processing file:", e)