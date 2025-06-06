import os
import librosa
import numpy as np
import noisereduce as nr
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


DATA_PATH = r"C:\Users\leduc\OneDrive\Desktop\bap tap uit\CS112\dataset"  
SAMPLE_RATE = 22050
NOISE_SAMPLE_DURATION = 0.5 


def extract_combined_features(file_path):
    try:

        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)


        noise_sample = y[:int(NOISE_SAMPLE_DURATION * sr)]

        # Reduce noise
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=1.0, stationary=True)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Chroma
        stft = np.abs(librosa.stft(y_denoised))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y_denoised)
        zcr_mean = np.mean(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y_denoised)
        rms_mean = np.mean(rms)


        features = np.hstack([
            mfcc_mean,
            chroma_mean,
            contrast_mean,
            [zcr_mean],
            [rms_mean]
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_dataset():
    features = []
    labels = []

    for label_name in ["dangerous", "non_dangerous"]:
        label_dir = os.path.join(DATA_PATH, label_name)
        label = 1 if label_name == "dangerous" else 0

        for file in tqdm(os.listdir(label_dir), desc=f"Loading {label_name}"):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(label_dir, file)
                feature_vector = extract_combined_features(file_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(label)

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    print("Extracting features with noise reduction...")
    X, y = load_dataset()

    print("\nDataset summary:")
    print(f"Feature shape: {X.shape}")
    print(f"Dangerous: {sum(y==1)} | Non-Dangerous: {sum(y==0)}")


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Dangerous", "Dangerous"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    joblib.dump(model, "animal_sound_classifier_with_denoise.pkl")
    print("\nModel saved: animal_sound_classifier_with_denoise.pkl")
