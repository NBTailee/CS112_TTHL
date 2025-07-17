import os
import librosa
import numpy as np
import noisereduce as nr
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


DATA_PATH = "data"  
SAMPLE_RATE = 22050
NOISE_SAMPLE_DURATION = 0.5  


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        noise = y[:int(NOISE_SAMPLE_DURATION * sr)]
        y_clean = nr.reduce_noise(y=y, sr=sr, y_noise=noise, stationary=True)

        mfcc = np.mean(librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=13).T, axis=0)
        stft = np.abs(librosa.stft(y_clean))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_clean))
        rms = np.mean(librosa.feature.rms(y=y_clean))

        return np.hstack([mfcc, chroma, contrast, [zcr], [rms]])
    except Exception as e:
        print(f"[!] Error processing {file_path}: {e}")
        return None


def load_dataset():
    X, y = [], []
    classes = [d for d in os.listdir(DATA_PATH)
               if os.path.isdir(os.path.join(DATA_PATH, d))]
    classes = sorted(classes, key=lambda s: s.lower())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print("Found classes:", classes)

    for cls in classes:
        dir_path = os.path.join(DATA_PATH, cls)
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(".wav")]
        print(f"📂 {cls} → {len(files)} files")

        for fn in tqdm(files, desc=f"Processing {cls}"):
            fv = extract_features(os.path.join(dir_path, fn))
            if fv is not None:
                X.append(fv)
                y.append(class_to_idx[cls])

    return np.array(X), np.array(y), classes, class_to_idx

if __name__ == "__main__":
    print("Loading and processing dataset...")
    X, y, classes, class_to_idx = load_dataset()

    if X.size == 0:
        raise RuntimeError("No data found. Check folder names and WAV files.")

    print(f"\n Dataset loaded: {X.shape[0]} samples, feature vector size = {X.shape[1]}")


    for cls in classes:
        count = np.sum(y == class_to_idx[cls])
        print(f"  - {cls}: {count} samples")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n Data split → Train: {len(y_train)}, Test: {len(y_test)}")


    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    
    joblib.dump({"model": clf, "classes": classes}, "animal_name_classifier.pkl")
    print("\n Model saved to: animal_name_classifier.pkl")
