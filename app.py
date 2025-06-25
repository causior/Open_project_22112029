import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import os

# Define custom attention layer (must match model)
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

# Load model
model = load_model("model.h5", custom_objects={"Attention": Attention})

# Emotion labels
class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# Extract log-mel spectrogram
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

# Prediction function
def predict_emotion(audio_path):
    mel = extract_log_mel(audio_path)
    mel = np.expand_dims(mel, axis=0)
    prediction = model.predict(mel)
    emotion = class_names[np.argmax(prediction)]
    return emotion

# Streamlit UI
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` file (preferably 3 seconds long) and get the predicted emotion.")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("Analyzing..."):
            emotion = predict_emotion("temp_audio.wav")
        st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")
    except Exception as e:
        st.error("Error processing audio file. Please upload a valid .wav file.")
        st.error(str(e))
    
    os.remove("temp_audio.wav")  

st.markdown("---")
st.markdown("<div style='text-align: center;'>Developed as a final year Deep Learning project</div>", unsafe_allow_html=True)
