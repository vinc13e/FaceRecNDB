import face_recognition
import cv2
import numpy as np
import json
import mysql.connector as mysql


import sys
name = sys.argv[1]
img = sys.argv[2]


limage = face_recognition.load_image_file(img)
face_features = face_recognition.face_encodings(limage)[0]
face_json = json.dumps(face_features.tolist())  


mydb = mysql.connect(
  host="host",
  user="user",
  password="password"
)

mycursor = mydb.cursor()
sql = "insert into facerec.faces(name,faceFeatures) values(%s, %s)"
val = (name, face_json)
mycursor.execute(sql, val)

mydb.commit()
print(mycursor.rowcount, "Record inserted.")
