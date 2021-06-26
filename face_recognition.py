# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np


#For a gven image, returns a rectangle for face detected with grayscale image
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)    #convert to gray image
    face_haar_cascade = cv2.CascadeClassifier('D:/Documents/Spyder/Face_recognition/HarrCascade/haarcascade_frontalface_default.xml') #loading Haar cascade file
    
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.35, minNeighbors=5)   #rectangle
    return faces, gray_img



#give labels to training images
def labels_for_training_images(directory):
    faces = []
    faceID = []
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print('Skipping the system file')
                continue
            
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)    #fetch image path
            print("image path: ", img_path)
            test_img = cv2.imread(img_path)    #load images one by one
            
            if test_img is None:
                print('Image is not loaded properly')
                continue
            
            faces_rect, gray_img = faceDetection(test_img)   #call faceDetection function to return face location in each images
            if len(faces_rect)!=1:    #assuming only one person in image
                continue 
                        
            (x,y,w,h) = faces_rect[0]      #rectangle coordinates
            roi_gray = gray_img[y:y+w, x:x+h]   #crop the region of interest 
            faces.append(roi_gray)
            faceID.append(int(id))
            
    return faces, faceID



#train the classifier using training images
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

#draw a bounding box around the face
def draw_rect(test_img,face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), thickness=5)
    
#write  the name of the person in the image
def put_text(test_img, text, x,y):
    cv2.putText(test_img,text,(x,y), cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)