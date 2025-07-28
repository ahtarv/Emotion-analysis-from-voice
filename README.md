# 🎙️ Emotion Analysis from Voice

A lightweight Python tool (130 LOC) that captures live microphone input and detects the speaker's emotion using audio features. Built with `librosa`, `scikit-learn`, and a pre-trained model on the RAVDESS dataset.

## 🚀 Features

- 🎧 Real-time microphone input
- 🔍 MFCC (Mel-Frequency Cepstral Coefficients) feature extraction
- 🧠 Emotion classification using Random Forest
- 📊 Trained on the [RAVDESS dataset](https://zenodo.org/record/1188976)
- 🛠 Simple and modular (130 lines of code)

---

## 🎯 Emotions Detected

- Happy
- Sad
- Angry
- Fearful
- Calm
- Neutral

> *(You can customize the label set based on the model and dataset.)*

---

## 🧰 Requirements

```bash
pip install librosa sounddevice numpy scikit-learn
