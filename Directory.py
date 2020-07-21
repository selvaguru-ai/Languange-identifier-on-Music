# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:10:24 2020

@author: Selva
"""
import os
import librosa
import math
import json

files_path = 'f:\\DataSet'
json_path = 'f:\\DataSet\mfcc_dataset.json'
DURATION = 30 #seconds
SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
##Creating a function that gets the MFCC and audio data value
def store_mfcc(files_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    #num_segments- chopping of each songs into number of segments in this case it is 5 
    #entries = os.listdir(files_path)
    #dictionary to store mapping,labels and mfcc
    
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
            }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    #go through all the folders present under f:\\DataSet
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(files_path)):
        
        #ensure we are not at the root level
        if dirpath is not files_path:
            dirpath_components = dirpath.split("/")
            #saving the folder names as the labels
            semantic_label = dirpath_components[-1]
            #Adding the mapping labels into the dictionary
            data["mapping"].append(semantic_label)
            print ("\nProcessing {}".format(semantic_label))
            #processing a specific language
            for f in filenames:
                
                #load audio file 
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                #divide the loaded signals into bunch of segments
                #Extract them to MFCC and store the data under data dictionary
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    
                    #extracting MFCC for a particular sample using the above start and finish sample values
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr = sr,
                                                n_fft = n_fft,
                                                n_mfcc = num_mfcc,
                                                hop_length = hop_length)
                    mfcc = mfcc.T
                    #ensure that there are same number of MFCC vectors for each tracks
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{},segment:{}".format(file_path, s+1))
    
    with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)
            
if __name__ == "__main__":
    store_mfcc(files_path, json_path, num_segments=10)
        
                
            
    


    
    