# Animal Sound Classifier

This project is an audio classification application that identifies animals based on their sounds. It uses a trained Random Forest model along with feature extraction from recorded audio, all wrapped in an interactive web interface built with Streamlit.

## Features

- Record sound from your microphone directly in the browser
- Preprocess audio with noise reduction
- Extract audio features (MFCC, Chroma, Spectral Contrast, etc.)
- Predict the animal name using a trained machine learning model
- Clean and simple user interface
- Optional region selector (decorative)


## Demo



https://github.com/user-attachments/assets/8ba5bec9-f1d4-4628-b350-ff24df3085dc



## Technologies Used

- Python 3.11
- Streamlit
- scikit-learn
- librosa
- noisereduce
- numpy
- sounddevice
- joblib

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```



### Run the App
```bash
streamlit run app.py
```

### Project Structure
```bash
.
├── app.py                       
├── animal_classifier.py        
├── animal_sound_classifier_with_denoise.pkl  
├── data/                      
└── README.md

```
