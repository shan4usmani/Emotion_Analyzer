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
import code1
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






        # label = ERS.Label(self.root, text="Emotion Analyzer", fg="red", font=("Arial Bold", 20), bg = "white")
        # label.grid(row=0, column=2, pady = 50, padx = 150)

        # loadButton = ERS.Button(self.root, text="Load Audio", command=self.loadAudio, bg="white", fg="red")
        # loadButton.grid(row=1, column=2, pady = 10, padx = 100)
        
        # recordButton = ERS.Button(self.root, text="Record Audio", command=self.recordAudio, bg="white", fg="red")
        # recordButton.grid(row=2, column=2, pady = 10, padx = 100)


        # playButton = ERS.Button(self.root, text="Play Audio", command = self.playAudio, bg="white", fg="red")
        # playButton.grid(row=3 , column=2, pady = 10, padx =100)

        # loadModel = ERS.Button(self.root, text="Load Model", command = self.loadModel, bg="white", fg="red")
        # loadModel.grid(row=4 , column=2, pady = 10, padx =100)

        # procButton = ERS.Button(self.root, text="Extract Features & Classify", command=self.processAudio, bg="white", fg="red")
        # procButton.grid(row=5, column=2, pady = 10, padx = 100)

        # showWaveButton = ERS.Button(self.root, text="Show Wave", command=self.showWave, bg="white", fg="red")
        # showWaveButton.grid(row=6, column=2, pady = 10, padx = 100)


        # showSpectrogramButton = ERS.Button(self.root, text="Show Graph", command=self.showSpectrogram, bg="white", fg="red")
        # showSpectrogramButton.grid(row=7, column=2, pady = 10, padx = 100)


        # exitButton = ERS.Button(self.root, text="Quit", command=self.quitProgram, bg="white", fg="red")
        # exitButton.grid(row=8, column=2, pady = 10, padx = 100)

        global var
        var = StringVar()
        label3 = ERS.Label(self.root, textvariable=var, fg="red", font=("Arial Bold", 16), bg = "white").place(x= 275, y = 350)
        # .place(x= 50, y = 250)
        # label3.grid(row=4, column=1, pady = 10, padx = 100)
        var.set("")


        self.root.mainloop()

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

        # file1 = pd.read_csv("features.csv" , skiprows = 0)

        # Features = pd.DataFrame(file1)
        # Features['labels'] = Y
        # Y = Features['labels'].values
        # print(Y)

        Features2 = pd.read_csv("features.csv")

        X = Features2.iloc[: ,:-1].values
        Y = Features2['labels'].values
        print(Y)
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

        print(Y)
        # encoder = OneHotEncoder()
        # Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()


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
        # print(y_pred)
        # id = 1  # Song ID
        # feature_set = pd.DataFrame()  # Feature Matrix

        # # Individual Feature Vectors
        # tempo_vector = pd.Series()
        # average_beats = pd.Series()
        # chroma_stft_mean = pd.Series()
        # chroma_cq_mean = pd.Series()
        # chroma_cens_mean = pd.Series()
        # mel_mean = pd.Series()
        # mfcc_mean = pd.Series()
        # mfcc_delta_mean = pd.Series()
        # rmse_value = pd.Series()
        # energy_value = pd.Series()
        # pow_value = pd.Series()


        #     # Reading Song

        # y, sr = librosa.load(file_path, duration=5)
        # S = np.abs(librosa.stft(y))

        # # Extracting Features
        # tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        # chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        # chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        # melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        # mfcc = librosa.feature.mfcc(y=y, sr=sr)
        # mfcc_delta = librosa.feature.delta(mfcc)
        # rmse = librosa.feature.rms(y=y)
        # energy = librosa.feature.melspectrogram(y=y, sr=sr, power=1)
        # power = librosa.feature.melspectrogram(y=y, sr=sr, power=2)
        # energy = librosa.core.amplitude_to_db(energy)
        # power = librosa.core.power_to_db(power)

        # # Transforming Features
        # tempo_vector.at[id]=tempo
        # average_beats.at[id]=np.average(beats)
        # chroma_stft_mean.at[id]= np.mean(chroma_stft)  # chroma stft
        # chroma_cq_mean.at[id] =np.mean(chroma_cq)  # chroma cq
        # chroma_cens_mean.at[id]= np.mean(chroma_cens)  # chroma cens
        # mel_mean.at[id]= np.mean(melspectrogram)  # melspectrogram
        # mfcc_mean.at[id]= np.mean(mfcc)  # mfcc
        # mfcc_delta_mean.at[id] =np.mean(mfcc_delta) # mfcc delta
        # rmse_value.at[id]= np.mean(rmse)
        # energy_value.at[id] =np.mean(energy)
        # pow_value.at[id] =np.mean(power)
        # print(file_path)

        # # Concatenating Features into one csv and json format
        # feature_set['tempo'] = tempo_vector  # tempo
        # feature_set['average_beats'] = average_beats
        # feature_set['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
        # feature_set['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
        # feature_set['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
        # feature_set['melspectrogram_mean'] = mel_mean  # melspectrogram
        # feature_set['mfcc_mean'] = mfcc_mean  # mfcc
        # feature_set['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
        # feature_set['rmse_value'] = rmse_value  # rmse
        # feature_set['energy_value'] = energy_value  # rmse
        # feature_set['pow_value'] = pow_value  # rmse



        # # Converting Dataframe into CSV Excel and JSON file
        # feature_set.to_csv('result.csv')
        # Result=code1.code("result.csv")

    
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
        # Audio(path)
        # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        # log_S = librosa.amplitude_to_db(S, ref=np.max)
        # plt.figure(figsize=(12, 4))
        # librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.title('Waveplot of input Audio')
        # plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()    

    def quitProgram(self):
        self.root.destroy()


LP = LoginPage()