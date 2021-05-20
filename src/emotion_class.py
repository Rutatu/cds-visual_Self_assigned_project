#!/usr/bin/env python



''' ---------------- About the script ----------------

Self-assigned project: Emotion recognition using transfer learning  

The script takes a face images FER-2013 dataset as an input, instantiates a pre-trained VGG-Face deep CNN model and adds a fully-connected classifier relevant to the new task on top. The new classifier´s weights are being updated during training. Lastly, the script outputs the classification report and performance graph.


Argument:
    -trd,     --train_data:      Directory of training data
    -vald,    --val_data:        Directory of validation data
    -optim,   --optimizer:       Method to update the weight parameters to minimize the loss function. Choose between SGD and Adam.
    -lr,      --learning_rate:   The amount that the weights are updated during training. Default = 0.001
    -ep,      --epochs:          Defines how many times the learning algorithm will work through the entire training dataset. Default = 30



Example:    
    
    with default values:
        $ python emotion_class.py -trd ../data/Face_emotions/train -vald ../data/Face_emotions/test -optim Adam
        
    with optional arguments:
        $ python emotion_class.py -trd ../data/Face_emotions/train -vald ../data/Face_emotions/test -optim SGD -lr 0.003 -ep 100


'''






"""---------------- Importing libraries ----------------
"""
# data tools
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd

sys.path.append(os.path.join(".."))

# Import pathlib
from pathlib import Path
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)

# VGG_Face model
from keras_vggface.vggface import VGGFace

# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     Dropout,
                                     ZeroPadding2D)

# generic model object
from tensorflow.keras.models import Model, Sequential

# optimizers
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from keras.utils import plot_model


# Command-line interface
import argparse


"""---------------- Main script ----------------
"""


def main():
    
    """------ Argparse parameters ------
    """
    # Instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser(description = "[INFO] Classify emotions and print out performance accuracy report")
    
    # Adding optional (with defaults) and required arguments
    parser.add_argument("-trd", "--train_data", required=True, help = "Directory of training data")
    parser.add_argument("-vald", "--val_data", required=True, help = "Directory of validation data")
    parser.add_argument("-optim", "--optimizer",required=True, help = "Method to update the weight parameters to minimize the loss function. Choose between SGD and Adam.")
    parser.add_argument("-lr", "--learning_rate", default = 0.001, type = float, help = "The amount that the weights are updated during training. Default = 0.001")
    parser.add_argument("-ep", "--epochs", default = 50, help = "Defines how many times the learning algorithm will work through the entire training dataset. Default = 50")


    # Parsing the arguments
    args = vars(parser.parse_args())
    
    # Saving parameters as variables
    trd = args["train_data"] # training data dir
    vald = args["val_data"] # validation data dir
    optim = args["optimizer"] # optimizer
    lr = args["learning_rate"] # learning rate
    ep = int(args["epochs"]) # epochs
    
    
    """------ Loading data and preprocessing ------
    """


    # getting training and validation data
    print("[INFO] loading and preprocessing training and validation data ...")
    train = get_data(os.path.join(trd))
    val = get_data(os.path.join(vald))
 
    
    #Create ouput folder, if it doesn´t exist already, for saving the classification report, performance graph and model´s architecture 
    if not os.path.exists("../output"):
        os.makedirs("../output")
    
    
    
    """------ Preparing training and validations sets ------
    """
       
    # empty lists for training and validation images and labels
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    # appending features (images as numpy arrays) and labels to the empty lists for further processing
    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    # normalizing the data (rescaling RGB channel values from 0-255 to 0-1)
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    # integers to one-hot vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.fit_transform(y_val)
    
    

    
    """------ Loading and modifying VGG-Face CNN model ------
    """
    # Loading VGG-Face model
    vggface = VGGFace()
    
    # Convolution Features (loading only feature extraction layers and adjusting the input shape)
    vgg_face = VGGFace(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights = 'vggface') # pooling: None, avg or max

    # Disabling the convolutional layers before training.
    # marking loaded layers as not trainable
    for layer in vgg_face.layers:
        layer.trainable = False
        
    # Adding new classifier layers
    flat1 = Flatten()(vgg_face.layers[-1].output)
    class1 = Dense(256, 
                   activation='relu')(flat1)

    #dropout = Dropout(0.5)(class1)

    output = Dense(7, 
                   activation='softmax')(class1)


    # Defining new model
    vgg_face = Model(inputs=vgg_face.inputs, 
                    outputs=output)
    
    # ploting and saving model´s architecture
    plot_model(vgg_face, to_file='../output/VGG-Face_CNN´s_architecture.png',
               show_shapes=True,
               show_dtype=True,
               show_layer_names=True)
    
    # Printing that model´s architecture graph has been saved
    print(f"\n[INFO] Model´s architecture graph has been saved")
    
    
    
    
        
    if optim == "Adam":
        opt = Adam(lr=lr)

        # Compile model
        vgg_face.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
        # train the model
        print("[INFO] training VGG-Face CNN model ...")
        H = vgg_face.fit(x_train, y_train, 
                      validation_data=(x_val, y_val), 
                      batch_size=128,
                      epochs=ep,
                      verbose=1)
        
      
    
    elif optim == "SGD":
    
        opt = SGD(lr=lr)

        # Compile model
        vgg_face.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
        # train the model
        print("[INFO] training VGG-Face CNN model ...")
        H = vgg_face.fit(x_train, y_train, 
                      validation_data=(x_val, y_val), 
                      batch_size=128,
                      epochs=ep,
                      verbose=1)
    
    
    else:
        print("Not a valid optimizer. Choose between 'SGD' and 'Adam'.")
    
 
    
    
    """------ VGG-Face CNN model´s output ------
    """
 
    # ploting and saving model´s performance graph
    plot_history(H,ep)
    
    # Printing that performance graph has been saved
    print(f"\n[INFO] Model´s performance graph has been saved")
    
        
    # Extracting the labels
    labels = os.listdir(os.path.join(trd))
    # Classification report
    predictions = vgg_face.predict(x_val, batch_size=32)
    print(classification_report(y_val.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels))
    
    
    
    # defining full filepath to save .csv file 
    outfile = os.path.join("../", "output", "Emotions_classifier_report.csv")
    
    # turning report into dataframe and saving as .csv
    report = pd.DataFrame(classification_report(y_val.argmax(axis=1), predictions.argmax(axis=1), target_names=labels, output_dict = True)).transpose()
    report.to_csv(outfile)
    print(f"\n[INFO] Classification report has been saved")
    
    print(f"\n[INFO] VGG-Face CNN model has been trained and evaluated successfully")


      
       
    
    
    
    
    
"""------ Functions ------
"""
  
        
        
            
# this function was developed for use in class and has been adapted for this project
def plot_history(H, epochs):
    '''
    visualize model´s performance: training and validation loss, 
    training and validation accuracy
    
    '''
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('../output/Emotions_classifier_performance.png')


    

# this function was taken from Gautam (2020) and adapted for this project    
def get_data(data_dir):
    '''
    loads training and validation data/images,
    preprocesses images and converts them into numpy array  
    
    '''
    # Extracting the labels
    labels = os.listdir(os.path.join(data_dir))
    # an empty list to store data
    data = [] 
    # a loop through each folder according to labels
    for label in labels: 
        # defining folder path
        path = os.path.join(data_dir, label)
        # assigning an index to a label
        class_num = labels.index(label)
        # a loop through each image in each folder in the path
        for img in os.listdir(path):   # returns a list containing the names of the entries in the directory given by path
            # convert the image pixels to a numpy array
            # reading an image into an object called 'image array'
            img_arr = cv2.imread(os.path.join(path, img))
            #image = img_to_array(img)
            # prepare the image for the VGG model
            #img_arr = preprocess_input(img_arr)
            data.append([img_arr, class_num])
    # return a numpy array             
    return np.array(data, dtype=object)
                
   

    
"""
"""
    
    
    
         
# Define behaviour when called from command line
if __name__=="__main__":
    main()
