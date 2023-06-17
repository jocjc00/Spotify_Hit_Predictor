# BCI3333 Machine Learning Application - Final Assessment

# Chaw Jo Chia CB20130

## Spotify Hit Predictor: dataset-of-10s.csv (https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset?select=dataset-of-10s.csv)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title and description
st.title('Spotify Hit Predictor')
st.write('A dataset containing 6398 tracks information and fetched using Spotify Web API .')

st.write("""
        Features: 
        - Danceability: float, ranging from 0.0 to 1.0, the suitability of the track for dancing
        - Energy: float, ranging from 0.0 to 1.0, the intensity and activity of the track
        - Loudness: float, ranging from -60.0 to 0.0, the overall loudness of the track in dB
        - Speechiness: float, ranging from 0.0 to 1.0, the presence of spoken words in the track
        - Acousticness: float, ranging from 0.0 to 1.0, the confidence of the track being acoustic
        - Instrumentalness: float, ranging from 0.0 to 1.0, the likelihood of the track being instrumental
        - Liveness: float, ranging from 0.0 to 1.0, the presence of a live audience in the track
        - Valence: float, ranging from 0.0 to 1.0, the musical positiveness. Higher values mean more positivity
        - Duration_ms: int, the duration of the track in milliseconds
        - Time_signature: int, the overall time signature of the track, indicating the number of beats in each bar
        - Chorus_hit: float, the duration of the chorus in the track
        """)


# Sidebar - User input features
st.sidebar.header('User Input Features')

# Collects user input features
def user_input_features():
    danceability = st.sidebar.slider('danceability', 0.0, 1.0, 1.0, 0.01)
    energy = st.sidebar.slider('energy', 0.0, 1.0, 0.9, 0.01)
    loudness = st.sidebar.slider('loudness', -60.0, 0.0, -22.5, 0.1)
    speechiness = st.sidebar.slider('speechiness', 0.0, 1.0, 0.6, 0.01)
    acousticness = st.sidebar.slider('acousticness', 0.0, 1.0, 0.8, 0.01)
    instrumentalness = st.sidebar.slider('instrumentalness', 0.0, 1.0, 0.5, 0.01)
    liveness = st.sidebar.slider('liveness', 0.0, 1.0, 0.6, 0.01)
    valence = st.sidebar.slider('valence', 0.0, 1.0, 0.9, 0.01)
    duration_ms = st.sidebar.slider('duration_ms', 0, 1000000, 600000, 1000)
    time_signature = st.sidebar.slider('time_signature', 0, 5, 4, 1)
    chorus_hit = st.sidebar.slider('chorus_hit', 0.0, 10.0, 6.0, 0.1)

    data = {
            'danceability': danceability,
            'energy': energy,
            'loudness': loudness,
            'speechiness': speechiness,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'valence': valence,
            'duration_ms': duration_ms,
            'time_signature': time_signature,
            'chorus_hit': chorus_hit,
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Read dataset
spotify_raw = pd.read_csv('dataset-of-10s.csv')

# Drop duplicate rows
spotify_raw.drop_duplicates(inplace=True)

# Drop rows with missing values
spotify_raw.dropna(inplace=True)

# Drop unnecessary columns
spotify = spotify_raw.drop(columns=['track', 'artist', 'uri', 'sections', 'tempo', 'mode', 'key', 'target'])

# Combine user input features and the dataset
df = pd.concat([input_df, spotify], axis=0)

# Select user input data at first row
df = df[:1]

# Load the saved model
load_clf = pickle.load(open('spotify_rf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

# Display prediction
st.subheader('Prediction')

# Predicted output: 0 = Flop, 1 = Hit
prediction_text = 'Flop' if prediction[0] == 0 else 'Hit'
st.write('The predicted track target is:', '**' + prediction_text + '**')
