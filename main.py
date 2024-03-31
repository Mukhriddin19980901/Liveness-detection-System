import math
from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import time

rec = cv2.VideoCapture(0)

rec.set(3,720)
rec.set(4,480)

Yolo = YOLO("../models/nano_vers_60.pt")
classnames = ['fake' ,'real']

prev_fr_time = 0
new_fr_time = 0

given_confidence = 0.6

while True:
    new_fr_time = time.time()
    _ , img = rec.read()
    result  = Yolo(img,stream=True,verbose=False)
    for i in result:
        boxes = i.boxes
        for j in boxes:
            x1,y1,x2,y2 = j.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1 ,y2-y1
            cvzone.cornerRect(img , (x1,y1,w,h))
            confidence=math.ceil((j.conf[0]*100))/100
            cls = int(j.cls[0])

            if confidence> given_confidence:

                if classnames[cls]=="real":
                    colour = (0,255,0)
                else:
                    colour = (0,0,255)
                cvzone.cornerRect(img,(x1,y1,w,h),colorC=colour,colorR=colour)
                cvzone.putTextRect(img,f"{classnames[cls].upper()} {int(confidence)*100}%" , (max(0,x1) , max(35,y1)) ,
                                   scale=2,thickness=4,colorR=colour,colorB=colour)

    FPS = 1/ (new_fr_time-prev_fr_time)
    prev_fr_time = new_fr_time
    print(FPS)
    cv2.imshow("Webcam" ,img)
    cv2.waitKey(1)