import numpy as np
import librosa
import joblib
import noisereduce as nr

SAMPLE_RATE = 22050
NOISE_SAMPLE_DURATION = 0.5  


data = joblib.load("/Users/tai.leduc/Desktop/bai tap uit/CS112_TTHL/animal_name_classifier.pkl")
model = data["model"]
classes = data["classes"]  

def extract_features_from_array(audio_array, sr=SAMPLE_RATE):
    try:
        noise_sample = audio_array[:int(NOISE_SAMPLE_DURATION * sr)]
        denoised = nr.reduce_noise(y=audio_array, sr=sr, y_noise=noise_sample, prop_decrease=1.0, stationary=True)

        mfcc = librosa.feature.mfcc(y=denoised, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        stft = np.abs(librosa.stft(denoised))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        zcr = librosa.feature.zero_crossing_rate(denoised)
        zcr_mean = np.mean(zcr)

        rms = librosa.feature.rms(y=denoised)
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
        print("Feature extraction error:", e)
        return None

def predict_from_audio(audio_array, sr=SAMPLE_RATE):
    features = extract_features_from_array(audio_array, sr)
    if features is None:
        return "Unknown"
    pred_idx = model.predict([features])[0]
    return classes[pred_idx]
