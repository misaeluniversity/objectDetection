# Instructions to use
# python3 (Object Detection Class) --protoFile (path for the prototxt) --modelFile (path for the model)

# Example
# python3 real_time_object_detection.py --protoFile ProtoTextConf.txt --modelFile TrainModel.caffemodel

########## Importing the necessary packages ##########

#Library with funtions for image handleing
import imutils

#Computer Vision Library
import cv2

#Speach Library
import pyttsx3 as se

#Library with scientific funtions
import numpy as np

#Library with  command-line interfaces funtions
import argparse as argp

#Library for image handleling
from PIL import Image

#Google AIY drivers
from aiy.board import Board

#Library for handleling PiCamera
from picamera import PiCamera

#Initialize Camera and set the resolution
camera = PiCamera(resolution = (320,240))

#Initialization of the voice engine
speakMotor = se.init()

########## Set properties of the voice engine ##########

# Speed percent (can go over 100)
speakMotor.setProperty('rate', 150)

# Volume 0-1
speakMotor.setProperty('volume', 0.9)  

#Setting language of the voice (language) engine to English
languages = speakMotor.getProperty('voices')

#Looping through the languages and selecting english
for language in languages:
    
    print ("[INFO] "+str(language)+"\n")
    
    if language.languages[0] == u'en_US':
        speakMotor.setProperty('voice', language.id)
        break

########## Parsing and Creation of initial arguments ##########

argumentsParsed = argp.ArgumentParser()

#Parsing argument for the prototxt file 
argumentsParsed.add_argument("-p", "--protoFile", required=True,
    help="path for the prototxt file that is going to be deploy")

#Parsing argument for the model file
argumentsParsed.add_argument("-m", "--modelFile", required=True,
    help="path for the file the contain the train model")

#Parsing argument for setting the confidence level
argumentsParsed.add_argument("-c", "--confidenceLevel", type=float, default=0.4,
    help="level of probability for the minimum detections")

#Setting dictionary of arguments for application
arguments = vars(argumentsParsed.parse_args())

########## Initializing the list of labels difine in the model ##########

TRAIN_LABELS = ["bus", "car", "background",
                "aeroplane", "bicycle", "bottle",
                "cat", "chair", "dog", "horse",
                "motorbike","cow", "diningtable",
                "person", "pottedplant", "sheep",
                "sofa", "bird", "boat", "train", "tvmonitor"]

########## Initializing train model ##########

print("[INFO] Initializing train model...\n")

trainNetwork = cv2.dnn.readNetFromCaffe(arguments["protoFile"], arguments["modelFile"])

print("[INFO] Model and libraries loaded...\n")


# loop over to process image
while True:
    #Wait for the button to trigger the camera and process the image
    with Board() as voiceBoard:
        
        print('[INFO] Press button to start recording.\n')
        voiceBoard.button.wait_for_press()
        listObjects = []
    
        #taking photo
        camera.capture('picture.jpg')

        print("[INFO] Loading image.\n")

        # Loading image
        img = cv2.imread("picture.jpg")

        image = cv2.imread("picture.jpg")
        
        image = imutils.resize(image, width=400)

        # Transforming the image to blob that the Deep Neural Network could read
        ImageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
            0.007843, (300, 300), 127.5)

        # Passing the blob through the Deep Neural Network
        trainNetwork.setInput(ImageBlob)
        
        # Getting the predictons from the Deep Neural Network based on the blob
        results = trainNetwork.forward()

        # Evaluating the results from the Deep Neural Network
        for i in np.arange(0, results.shape[2]):
            # Collecting the cofidence level of the elements detected by Deep Neural Network
            confidenceLevel = results[0, 0, i, 2]

            # Selecting the results that meet the confidence level
            if confidenceLevel > arguments["confidenceLevel"]:
                # Selecting the index of the labels from the predictions 
                idx = int(results[0, 0, i, 1])
                
                # Appending labels of detected elements
                listObjects.append(TRAIN_LABELS[idx])                  

        #Say object detected
        for element in listObjects:
            print ("[INFO] "+element+"\n")
            speakMotor.say(element)
            speakMotor.runAndWait()
        if len(listObjects) == 0:
            print ("[INFO] No objects Detected.\n")
            speakMotor.say("No objects Detected")
            speakMotor.runAndWait()
