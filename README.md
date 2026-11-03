# Real-Time Face Recognition Attendance System

An automated smart attendance system built with Python, OpenCV, and the face_recognition library. It captures live video from a webcam, detects known faces, and automatically logs attendance with the person's name, time, and date into a CSV file.

## Features
- Real-time face detection and recognition using deep learning encodings.
- Automatic entry logging (Name, Time, Date) to attendance.csv.
- Visual feedback with bounding boxes and names on the video feed.

## Tech Stack
- Python
- OpenCV
- face_recognition
- Pandas

## How to Run
1. Place face images inside the dataset/ folder (e.g., PersonName.jpg).
2. Run the script:
   python face_attendance_system.py
3. Press Esc to exit.
