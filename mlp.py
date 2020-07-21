# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:03:55 2020

@author: Selva
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = 'f:\\DataSet\mfcc_dataset.json'
SAVED_MODEL_PATH = 'f:\\DataSet\model.h5'
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
    #Convert the JSON data list to a numpy array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

if __name__ == "__main__":
    #load_data
    inputs, targets = load_data(DATASET_PATH)
    
    # Split the data into train and test
    input_train, input_test, target_train, target_test = train_test_split(inputs, 
                                                                          targets,
                                                                          test_size=0.3)
    #Building the MLP architecture using Keras
    model = keras.Sequential([
        #input_layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        
        # 1st hidden layer a dense layer with relu activation to avoid vanishing gradients
        keras.layers.Dense(512, activation='relu'),
        #2nd hidden layer a dense layer with 256 neurons
        keras.layers.Dense(256, activation='relu'),
        #3rd hidden layer a dense layer with 64 neurons
        keras.layers.Dense(64, activation='relu'),
        
        #output layer it is a binary classification layer hence it will have only two outputs (English/Tamil -> 0/1)
        keras.layers.Dense(2, activation="softmax")
        ])
    
    #compiling the network
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()
    
    #training the network
    model.fit(input_train, target_train, validation_data=(input_test, target_test)
              ,epochs=50
              ,batch_size=32)
    
    #evaluate the model
    test_error, test_accuracy = model.evaluate(input_test, target_test)
    print(f"Test Error: {test_error}, Test_accuracy:{test_accuracy}")
    
    #save the model
    model.save(SAVED_MODEL_PATH)
    print("Saved model to the system")