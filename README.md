# 🎧 Speech Emotion Recognition Using Deep Learning

This project focuses on recognizing human emotions from short audio clips of **speech and song** using a deep learning model. The system is trained on emotion-labeled `.wav` audio and can predict one of 8 emotions from an uploaded file.

---

## 📌 Project Description

Human emotions such as happiness, sadness, anger, and fear are essential in communication. This project uses a **CNN-BiLSTM-Attention model** to classify emotions from audio data by analyzing patterns in **log-Mel spectrograms** extracted from `.wav` files. The system is deployed with a user-friendly web interface using **Gradio**.

---

## 🔍 Dataset and Preprocessing

### 📁 Dataset Source
The dataset consists of `.wav` files categorized by emotion and actor. It includes two main categories:
- **Speech Audio**
- **Song Audio**

Each filename encodes emotion ID, actor ID, and more.

### 🧪 Preprocessing Steps
1. **Audio Loading**: Using `librosa` at 22050 Hz, limited to 3 seconds.
2. **Log-Mel Spectrograms**: 128 Mel bands × 128 time steps.
3. **Zero Padding**: If audio < 3s, it's padded to match required length.
4. **Normalization**: Feature-wise mean subtraction and division by std.
5. **Data Augmentation**:
   - Add random Gaussian noise
   - Apply time masking on spectrogram
6. **Label Encoding**: Emotions one-hot encoded (8 classes)
7. **Train-Test Split**: 80% training, 20% testing (stratified)

---

## 🧠 Model Architecture

> Built using **TensorFlow Keras**

```text
Input: (128, 128, 1) Log-Mel Spectrogram

→ Conv2D → MaxPooling → BatchNorm → Dropout  
→ Conv2D → MaxPooling → BatchNorm → Dropout  
→ Conv2D → MaxPooling → BatchNorm → Dropout  
→ Reshape → BiLSTM → Attention  
→ Dense Softmax (8-class output)
