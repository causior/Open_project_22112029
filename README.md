# 🎙️ Speech Emotion Recognition Using Deep Learning

This project focuses on recognizing human emotions from short audio clips of **speech and song** using a deep learning model. The system is trained on emotion-labeled `.wav` audio files and can predict one of **8 emotions** from an uploaded file using a web interface built with **Gradio**.

---

## 📌 Project Description

Human emotions such as happiness, sadness, anger, and fear are essential in communication. This project uses a **CNN-BiLSTM-Attention model** to classify emotions from audio data by analyzing patterns in **log-Mel spectrograms** extracted from `.wav` files.

---

## 🎵 Dataset and Preprocessing

### 📁 Dataset Source
The dataset consists of `.wav` files categorized by:
- **Speech Audio**
- **Song Audio**

Each filename encodes metadata such as emotion ID and actor ID.

### ⚙️ Preprocessing Steps
1. **Audio Loading**: Using `librosa` at 22050 Hz, trimmed/padded to 3 seconds.
2. **Log-Mel Spectrograms**: 128 Mel bands × 128 time frames.
3. **Zero Padding**: Applied to audio clips < 3 seconds.
4. **Normalization**: Standardization (zero mean, unit variance).
5. **Data Augmentation**:
   - Gaussian noise
   - Time masking
6. **Label Encoding**: One-hot encoded labels for 8 emotions.
7. **Train-Test Split**: 80% training, 20% testing (stratified split).

---

## 🧠 Model Architecture

> Implemented using **TensorFlow Keras**



---

## 📊 Performance

### 🧾 Classification Report

| Emotion   | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Neutral   | 0.86      | 0.95   | 0.90     | 38      |
| Calm      | 0.94      | 0.96   | 0.95     | 75      |
| Happy     | 0.85      | 0.77   | 0.81     | 75      |
| Sad       | 0.82      | 0.72   | 0.77     | 75      |
| Angry     | 0.86      | 0.85   | 0.85     | 75      |
| Fear      | 0.81      | 0.85   | 0.83     | 75      |
| Disgust   | 0.80      | 0.82   | 0.81     | 39      |
| Surprise  | 0.77      | 0.85   | 0.80     | 39      |

### ✅ Overall Metrics
- **Accuracy**: 0.84  
- **Macro F1-score**: 0.84  
- **Weighted F1-score**: 0.84

---

## 🚀 How to Run

### 🧰 Requirements
Install the required packages:

```bash
pip install tensorflow librosa gradio numpy matplotlib scikit-learn
