import streamlit as st
import sounddevice as sd
import numpy as np
from animal_classifier import predict_from_audio

SAMPLE_RATE = 22050
DURATION = 3

st.title("Animal Sound Classifier")
st.write("Click the button below to record sound from your microphone and classify the animal.")


region = st.selectbox(
    "Select your region:",
    ["North America", "South America", "Europe", "Asia", "Africa", "Australia", "Antarctica"]
)

if st.button("Record and Classify"):
    st.info(f"Recording for {DURATION} seconds...")
    try:
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        st.success("Recording complete. Classifying...")

        prediction = predict_from_audio(audio, sr=SAMPLE_RATE)
        st.markdown(f"### Detected animal: **{prediction}**")

    except Exception as e:
        st.error(f"Error during recording or prediction: {e}")
