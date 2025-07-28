# 🎙️ Emotion Detection from Voice

A Python tool that detects human emotions from real-time speech input using MFCC feature extraction and a trained `RandomForestClassifier`. Supports both a command-line interface and a desktop GUI built with Tkinter.

---

## 🧠 What It Does

- Captures live audio from your microphone
- Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from speech
- Classifies the emotion using a Random Forest model trained on the [RAVDESS dataset](https://zenodo.org/record/1188976)
- Displays the predicted emotion as text and emoji 😄😢😠😲

---

## 🎯 Supported Emotions

| Code | Emotion    | Emoji |
|------|------------|-------|
| 01   | Neutral    | 😐    |
| 02   | Calm       | 😌    |
| 03   | Happy      | 😄    |
| 04   | Sad        | 😢    |
| 05   | Angry      | 😠    |
| 06   | Fearful    | 😱    |
| 07   | Disgust    | 🤢    |
| 08   | Surprised  | 😲    |

---

## 🛠️ Requirements

Install the necessary Python libraries:

```bash
pip install numpy librosa scikit-learn sounddevice

