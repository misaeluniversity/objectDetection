# USAGE
# python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3
from PIL import Image
from aiy.board import Board
from picamera import PiCamera



#Initialize Camera and set the resolution
camera = PiCamera(resolution = (320,240))

#Initialization of the voice engine
engine = pyttsx3.init()

# Set properties of the voice engine
engine.setProperty('rate', 150)    # Speed percent (can go over 100)
engine.setProperty('volume', 0.9)  # Volume 0-1

#Set language of the voice engine to English
voices = engine.getProperty('voices')
for voice in voices:
    print (voice)
    if voice.languages[0] == u'en_US':
        engine.setProperty('voice', voice.id)
        break

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("Model and libraries loaded...")


# loop over to process image
while True:
    #Wait for the button to trigger the camera and process the image
    with Board() as board:
        print('Press button to start recording.')
        board.button.wait_for_press()
        listObjects = []
    

        #taking photo
        camera.capture('room_ser.jpg')

        print("star loading image")

        # Loading image
        img = cv2.imread("room_ser.jpg")


        frame = cv2.imread("room_ser.jpg") 
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                    
                listObjects.append(CLASSES[idx])
                    
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        #Say object detected
        for element in listObjects:
            print (element)
            engine.say(element)
            engine.runAndWait()
        if len(listObjects) == 0:
            print ("No objects Detected")
            engine.say("No objects Detected")
            engine.runAndWait()

        # show the output frame
        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        #fps.update()

# stop the timer and display FPS information
#fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()