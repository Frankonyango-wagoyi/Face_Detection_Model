import cv2
import numpy as np
import os
from PIL import Image

cam=cv2.VideoCapture()
cam.set(3,640) #width
cam.set(4,480) #height

face_detector=cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

face_id=input('\n Enter user id and press enter')

print('\n[INFO] Initializing face capture.')

count=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.3,5)

for(x,y,w,h) in faces:
    cv2.rectangle(img(x,y),(x+w+y+h),(255,0,0),2)
    count+=1

    cv2.imwrite("Dataset/User." +str(face_id) +'.' +str(count) +'.jpg',gray[y:y+h ,x:x+h])
    cv2.imshow(image,img)

k=cv2.waitKey(100) & 0xff


if k ==27:
    raise Exception("K must not be 27")
elif count>=30:
    raise Exception("Count should not be greater than or equal to 30")

print("\n [INFO] Exiting program")
cam.release()
cv2.destroyAllWindows()





