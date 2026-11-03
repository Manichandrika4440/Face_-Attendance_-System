"""
Face Recognition Attendance System for Employees
Requirements:
pip install opencv-python face-recognition numpy pandas

Folder structure:
project/
   dataset/        (store employee images here)
   trainer/        (trained model will be saved)
   attendance.csv  (auto created)
   face_attendance.py

Place employee images in dataset folder named like: EmployeeName.jpg
"""
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime

path = 'dataset'
images = []
classNames = []

for img_name in os.listdir(path):
    img = cv2.imread(f'{path}/{img_name}')
    images.append(img)
    classNames.append(os.path.splitext(img_name)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    file_name = "attendance.csv"
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("Name,Time,Date\n")

    with open(file_name, "r+") as f:
        lines = f.readlines()
        nameList = [line.split(',')[0] for line in lines]

        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            date = now.strftime('%Y-%m-%d')
            f.writelines(f'\n{name},{time},{date}')

print("Encoding images...")
encodeListKnown = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        import numpy as np
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow("Face Recognition Attendance System",img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
