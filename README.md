#  Speech Emotion Recognition Using Deep Learning

This project focuses on recognizing human emotions from short audio clips of **speech and song** using a deep learning model. The system is trained on emotion-labeled `.wav` audio and can predict one of 8 emotions from an uploaded file.

---

##  Project Description

Human emotions such as happiness, sadness, anger, and fear are essential in communication. This project uses a **CNN-BiLSTM-Attention model** to classify emotions from audio data by analyzing patterns in **log-Mel spectrograms** extracted from `.wav` files. The system is deployed with a user-friendly web interface using **Gradio**.

---

##  Dataset and Preprocessing

###  Dataset Source
The dataset consists of `.wav` files categorized by emotion and actor. It includes two main categories:
- **Speech Audio**
- **Song Audio**

Each filename encodes emotion ID, actor ID, and more.

###  Preprocessing Steps
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

##  Model Architecture

> Built using **TensorFlow Keras**

```text
**Input**: (128, 128, 1) Log-Mel Spectrogram

→ **Conv2D** → **MaxPooling** → **BatchNormalization** → **Dropout**  
→ **Conv2D** → **MaxPooling** → **BatchNormalization** → **Dropout**  
→ **Conv2D** → **MaxPooling** → **BatchNormalization** → **Dropout**  
→ **Reshape** → **BiLSTM** → **Attention**  
→ **Dense (Softmax, 8-class output)**


---

##  Classification Report

| Emotion   | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| **Neutral**   | 0.86      | 0.95   | 0.90     | 38      |
| **Calm**      | 0.94      | 0.96   | 0.95     | 75      |
| **Happy**     | 0.85      | 0.77   | 0.81     | 75      |
| **Sad**       | 0.82      | 0.72   | 0.77     | 75      |
| **Angry**     | 0.86      | 0.85   | 0.85     | 75      |
| **Fear**      | 0.81      | 0.85   | 0.83     | 75      |
| **Disgust**   | 0.80      | 0.82   | 0.81     | 39      |
| **Surprise**  | 0.77      | 0.85   | 0.80     | 39      |

---

##  Overall Metrics

- **Accuracy**: **0.84**
- **Macro Average F1-score**: **0.84**
- **Weighted Average F1-score**: **0.84**

---

##  How to Run

### Requirements
- Python 3.8+
- TensorFlow
- Librosa
- Gradio
