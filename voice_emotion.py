import os
import librosa
import numpy as np
import sounddevice as sd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import tkinter as tk
from tkinter import messagebox
import threading 
import warnings

warnings.filterwarnings('ignore')

# --- Emotion & Emoji Maps ---
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'  # fixed typo: was "suprised"
}

emoji_map = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😄',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😱',
    'disgust': '🤢',
    'surprised': '😲'
}

# --- Feature Extraction ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# --- Dataset Loader ---
def load_dataset(dataset_path):
    X, y = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                emotion = emotion_map.get(emotion_code)
                if emotion:
                    file_path = os.path.join(root, file)
                    try:
                        features = extract_features(file_path)
                        X.append(features)
                        y.append(emotion)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
    return np.array(X), np.array(y)

# --- Record Live Audio ---
def record_audio(duration=2, sr=22050):
    print("🎙️ Speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def predict_live_cli(model):
    while True:
        print("Speak now..")
        audio = record_audio()
        mfcc = librosa.feature.mfcc(y=audio, sr = 22050, n_mfcc = 40)
        mfcc_mean = np.mean(mfcc.T, axis =0).reshape(1,-1)
        pred = model.predict(mfcc_mean)[0]
        print(f"Predicted Emotion: {emoji_map.get(pred, '?')} {pred}\n")

def predict_from_gui():
    status_label.config(text = "Recording...")
    result_label.config(text = "Listening...")
    root.update()

    audio = record_audio()
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1,-1)

    pred = model.predict(mfcc_mean)[0]
    emoji = emoji_map.get(pred,'?')

    result_label.config(text=f"{emoji} {pred}")
    status_label.config(text = "Tap to seapk again")

def launch_gui():
    global root, result_label, status_label

    root = tk.Tk()
    root.title("Emotion Detector")
    root.geometry("400x250")
    root.config(bg = "#121212")

    title = tk.Label(root, text = "Real-Time Emotion Detector", font = ("Helvetica", 16), fg = "white", bg = "#121212")
    title.pack(pady=10)

    result_label = tk.Label(root, text = "Ready", font = ("Helvetica", 24), fg = "cyan", bg = "#121212")
    result_label.pack(pady=10)

    status_label = tk.Label(root, text = "Tap to record emotion", font = ("Helvetica", 12),fg = "gray", bg = "#121212")
    status_label.pack(pady=5)

    record_btn = tk.Button(root, text = "Record", command = lambda: threading.Thread(target = predict_from_gui).start(),font = ("Helvetica", 14), bg = "green",fg = "white", padx = 10, pady = 5)
    record_btn.pack(pady=20)

    root.mainloop()
    
# --- Main ---
if __name__ == "__main__":
    print("📦 Loading and training on RAVDESS dataset...")

    dataset_path = "/home/atharvkp/Desktop/archive (2)"  # ✅ use full path here
    X, y = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("✅ Model trained.")
    print(classification_report(y_test, model.predict(X_test)))

    mode = input("\n Type 'cli' for terminal mode or 'gui' for windowed mode:").strip().lower()
    if mode == 'gui':
        launch_gui()
    else:
        predict_live_cli(model)
