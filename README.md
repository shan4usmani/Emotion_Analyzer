# Emotion_Analysis

Background Image Credits: Vecteezy.com

Datasets :

Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) : https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D) : https://www.kaggle.com/ejlok1/cremad
Toronto emotional speech set (TESS) : https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess
speech emotion annotated data for emotion recognition systems : https://www.kaggle.com/barelydedicated/savee-database


This aim of this project is to develop a deep learning model in order to detect emotions from audio files. this can be used in various applications. the future aim is to use this project in order to detect child abuse by providing the audio/speech. 

How to run: 

1)open and run Emotion_analyzer.ipynb. this file will import the datasets. visualize, preprocess the data and extract the features from the data. then it will create a CNN model and train it. finally it will save the model in your local directory. 

2)run the file GUInew.py. this will open up a GUI. the "add model" button will load the model. provide an audio for classification by clicking on "Load Audio" button. then click on classify to display the predicted emotion. 


