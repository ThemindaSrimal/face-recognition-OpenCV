# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import face_recognition as fr

test_img = cv2.imread('D:/Documents/Spyder/Face_recognition/TestImages/test.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
print('face detected:', faces_detected)

#comment thes when you are running the code from the second time
#faces, faceID = fr.labels_for_training_images('D:/Documents/Spyder/Face_recognition/TrainingImages')
#face_recognizer = fr.train_classifier(faces, faceID)
#face_recognizer.write('D:/Documents/Spyder/Face_recognition/trainingData.yml')


#uncomment these while running the code from second time onwards
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('D:/Documents/Spyder/Face_recognition/trainingData.yml')

name = {0:'Person 1 name'}  #create dictionary containing names for label

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)  #predict the label of the image
    
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    
    if (confidence>35):  #if confidence is greater than 35, don't print the name
        continue
    
    fr.put_text(test_img, predicted_name, x, y)
    
    print('Confidence:', confidence)
    print('label', label)
    
    resized_img = cv2.resize(test_img,(500,700))
    cv2.imshow('face detection', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
