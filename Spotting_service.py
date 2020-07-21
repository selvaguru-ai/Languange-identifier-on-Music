# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 20:02:26 2020

@author: Selva
"""
import tensorflow.keras as keras
import librosa
import numpy as np
import math

MODEL_PATH = 'f:\\DataSet\model.h5'
NUM_SAMPLES_TO_CONSIDER = 22050  # for 1 second
DURATION = 30 #seconds
SAMPLES_PER_TRACK = NUM_SAMPLES_TO_CONSIDER * DURATION
MFCC_final =[]
#singleton class
class _Spotting_service:
    
    model = None

    _mappings = [
        "English",
        "Tamil"]
    
    _instance = None
    
    def predict(self, file_path):
        print(file_path)
        #this is where the input data to the model in real time is fed into
        #extract MFCC of the loaded .mp3 file
        MFCCs = self.preprocess(file_path) # (# segments, # Coefficients)
        
        #convert 2d MFCCs array into 4d array -> (#samples, #segments, #coefficients, #channels)
        #MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        
        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]
        return predicted_keyword
    
    
    def preprocess (self, file_path, n_mfcc=13, n_fft=2048, hop_length = 512, num_segments=10):
        data = {
                "mfcc": []
            }
        #load the audio file
        signal, sr = librosa.load(file_path)
        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
         #divide the loaded signals into bunch of segments
                #Extract them to MFCC and store the data under data dictionary
        for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment
        
                #extract MFCCs
                MFCCs = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr = sr,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)
        
        
                MFCCs = MFCCs.T
                if len(MFCCs) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(MFCCs.tolist())
        return data["mfcc"]
def Spotting_service():
    #ensure we have only one instance
    if _Spotting_service._instance is None:
        _Spotting_service._instance = _Spotting_service()
        _Spotting_service.model = keras.models.load_model(MODEL_PATH)
    return _Spotting_service._instance

if __name__ == "__main__":
    
    kss = Spotting_service()
    keyword1 = kss.predict("Nallai-Allai.mp3")
    keyword2 = kss.predict("nila.mp3")
    
    print (f"Predicted language of the song Nallai-Allai is: {keyword1}")
    print(f"Predicted language of the song nila.mp3, {keyword2}")
        

