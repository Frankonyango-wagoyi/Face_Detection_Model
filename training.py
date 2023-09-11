import cv2
import numpy as np 
from PIL import Image 
import os 

path="Dataset"

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

def getImagesAndLabels(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path) ]
    facesamples=[]
    ids=[]

    for imagePath in imagePaths:
        PIL_img=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img,'uint8')

        id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(img_numpy)
        
        for (x,y,w,h) in faces:
            facesamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return facesamples.ids

print("\n [INFO] Training faces....")
faces,ids=getImagesAndLabels(path)
recognizer.train(faces,np.array(ids))

recognizer.write('trainer/trainer.yml')
print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))


