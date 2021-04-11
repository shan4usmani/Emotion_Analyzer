import sys
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
import tkinter as ERS
import winsound
import librosa
import librosa.display
import pandas as pd
file = "outputF.csv"
import pyaudio
import IPython.display as ipd
import scipy.io.wavfile as wav
import code
import warnings; 
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np

import os
import sys

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

Result="?"

class LoginPage():
    def __init__(self):
        self.root = ERS.Tk()
        self.root.geometry("500x400")

        image = PhotoImage(file='backnew.png')
        background=Label(self.root, image=image)
        background.place(x=0,y=0,relwidth=1, relheight=1)
        # self.root.geometry("1000x700")

        loadButton = ERS.Button(self.root, text="Load Audio",command=self.loadAudio, bg="black", fg="white", height = 2, width = 10).place(x= 15, y = 150)
        # loadButton.grid(row=3, column=1, pady = 10, padx = 80)
        
        recordButton = ERS.Button(self.root, text="Record Audio", bg="black", fg="white", height = 2, width = 10).place(x= 150, y = 150)
        # recordButton.grid(row=3, column=2, pady = 10, padx = 80)


        playButton = ERS.Button(self.root, text="Play Audio", command = self.playAudio, bg="black", fg="white", height = 2, width = 10).place(x= 275, y = 150)
        # playButton.grid(row=3 , column=3, pady = 10, padx =80)

        loadModel = ERS.Button(self.root, text="Load Model",command = self.loadModel, bg="black", fg="white", height = 2, width = 10).place(x= 400, y = 150)
        # loadModel.grid(row=5 , column=1, pady = 10, padx =80)

        procButton = ERS.Button(self.root, text="Classify",command=self.processAudio, bg="black", fg="white", height = 2, width = 10).place(x= 15, y = 250)
        # procButton.grid(row=5, column=2, pady = 10, padx = 80)

        showWaveButton = ERS.Button(self.root, text="Wave",command=self.showWave,  bg="black", fg="white", height = 2, width = 10).place(x= 150, y = 250)
        # showWaveButton.grid(row=5, column=3, pady = 10, padx = 80)


        showSpectrogramButton = ERS.Button(self.root, text="Spectrogram", command=self.showSpectrogram, bg="black", fg="white", height = 2, width = 10).place(x= 275, y = 250)
        # showSpectrogramButton.grid(row=7, column=2, pady = 10, padx = 80)


        exitButton = ERS.Button(self.root, text="Quit",command=self.quitProgram,  bg="black", fg="white", height = 2, width = 10).place(x= 400, y = 250)
        # exitButton.grid(row=8, column=2, pady = 10, padx = 80)

        label1 = ERS.Label(self.root, text="Predicted Emotion is :", fg="red", font=("Arial Bold", 16), bg = "white").place(x = 25, y = 350)
        # label.grid(row=0, column=2, pady = 50, padx = 150)

        credit = ERS.Button(self.root, text="Credits",command=self.credits, fg="red", font=("Arial Bold", 8), bg = "white").place(x = 430, y = 350)



        global var
        var = StringVar()
        label3 = ERS.Label(self.root, textvariable=var, fg="red", font=("Arial Bold", 16), bg = "white").place(x= 275, y = 350)
        # label3.grid(row=4, column=1, pady = 10, padx = 100)
        var.set("")


        self.root.mainloop()

    def credits(self): 
        self.root = ERS.Tk()
        self.root.geometry("500x400")
        label1 = ERS.Label(self.root, text="Credits", fg="red", font=("Arial Bold", 16), bg = "white").place(x = 230, y = 5)
        label2 = ERS.Label(self.root, text="Background Image : vecteezy.com", fg="red", font=("Arial Bold", 16), bg = "white").place(x = 25, y = 50)
        label3 = ERS.Label(self.root, text="Backend : Usama", fg="red", font=("Arial Bold", 16), bg = "white").place(x = 25, y = 100)
        label4 = ERS.Label(self.root, text="Frontend : Shan Usmani & Muneeba", fg="red", font=("Arial Bold", 16), bg = "white").place(x = 25, y = 150)
        label5 = ERS.Label(self.root, text="PI : Dr Najeed Ahmed ", fg="red", font=("Arial Bold", 16), bg = "white").place(x = 25, y = 200)
        label6 = ERS.Label(self.root, text="Co PI : Moona Kanwal ", fg="red", font=("Arial Bold", 16), bg = "white").place(x = 25, y = 250)
    def loadAudio(self):
        print("\t\t\t\t --Selecting File ..")
        global file_path
        file_path = filedialog.askopenfilename()
        print("\t --Loading Audio: " + file_path)
        return file_path

        # Need to call file_path somewhere so that it becomes global and accessible by every method

    def playAudio(self):
        print("\t\t\t\t --Playing File ..")
        winsound.PlaySound(file_path, winsound.SND_ASYNC)

    def loadModel(self):
        print("\t\t\t\t --Loading Model ..")
        # load model
        global model
        model = load_model('model.h5')
        # summarize model.
        model.summary()

        
    def recordAudio(self):
        RATE = 16000
        RECORD_SECONDS = 2.5
        CHUNKSIZE = 1024

        # initialize portaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
        print("***Recording ***")

        frames = []
        for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
            data = stream.read(CHUNKSIZE)
            frames.append(np.fromstring(data, dtype=np.int16))

        # Convert the list of numpy-arrays into a 1D array (column-wise)
        numpydata = np.hstack(frames)
        print("* done")
        # close stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        ipd.Audio(numpydata, rate=RATE)

        dir = "testingData"
        filename = "\output.wav"
        wav.write(dir + filename, RATE, numpydata)

    def processAudio(self):

        # taking any example and checking for techniques.
        # path = np.array(data_path.Path)[1]
        data, sample_rate = librosa.load(file_path)

        def noise(data):
            noise_amp = 0.035*np.random.uniform()*np.amax(data)
            data = data + noise_amp*np.random.normal(size=data.shape[0])
            return data

        def stretch(data, rate=0.8):
            return librosa.effects.time_stretch(data, rate)

        def shift(data):
            shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
            return np.roll(data, shift_range)

        def pitch(data, sampling_rate, pitch_factor=0.7):
            return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

        
        def extract_features(data):
            # ZCR
            result = np.array([])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
            result=np.hstack((result, zcr)) # stacking horizontally

            # Chroma_stft
            stft = np.abs(librosa.stft(data))
            chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_stft)) # stacking horizontally

            # MFCC
            mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mfcc)) # stacking horizontally

            # Root Mean Square Value
            rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
            result = np.hstack((result, rms)) # stacking horizontally

            # MelSpectogram
            mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel)) # stacking horizontally
            
            return result

        def get_features(path):
            data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
            
            # without augmentation
            res1 = extract_features(data)
            result = np.array(res1)
            
            # data with noise
            noise_data = noise(data)
            res2 = extract_features(noise_data)
            result = np.vstack((result, res2)) # stacking vertically
            
            # data with stretching and pitching
            new_data = stretch(data)
            data_stretch_pitch = pitch(new_data, sample_rate)
            res3 = extract_features(data_stretch_pitch)
            result = np.vstack((result, res3)) # stacking vertically
            
            return result


        X, Y = [], []
        # for path, emotion in zip(data_path.Path, data_path.Emotions):
        feature = get_features(file_path)
        for ele in feature:
            X.append(ele)
            # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
            # Y.append(emotion)

        print(X)

        Features = pd.DataFrame(X)
        # Features['labels'] = Y
        # Features.to_csv('features.csv', index=False)
        print(Features.head())
        X1 = Features.iloc[: ,:].values


        scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(X1)
        print(x_test.shape)

        x_test = np.expand_dims(x_test, axis=2)
        print(x_test.shape)

        
        Features2 = pd.read_csv("features.csv")

        X = Features2.iloc[: ,:-1].values
        Y = Features2['labels'].values
        print(Y)
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

        print(Y)
     

        pred_test = model.predict(x_test)
        print(pred_test)


        y_pred = encoder.inverse_transform(pred_test)
        print(y_pred)
        print("y_pred[1][0] = ", y_pred[1][0])
        t = y_pred[1][0]
        # if (y_pred[2][0] != "fear") or (y_pred[2][0] !="happy") or (y_pred[2][0] != "sad"):
        if t == "fear":
            var.set(t)
        
        elif t == "angry":
            var.set(t)
        
        elif t == "neutral":
            var.set(t)

        else :
            var.set("Unknown")
        

    
    def showSpectrogram(self):

        y, sr = librosa.load(file_path)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        log_S = librosa.amplitude_to_db(S, ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()

    def showWave(self):

        y, sr = librosa.load(file_path)

        plt.figure(figsize=(14,4))
        librosa.display.waveplot(y=y, sr=sr)
      
        plt.title('Waveplot of input Audio')
        # plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()    

    def quitProgram(self):
        self.root.destroy()


LP = LoginPage()