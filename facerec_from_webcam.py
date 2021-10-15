import face_recognition
import cv2
import numpy as np
import json
import mysql.connector
import threading
import time
from threading import Thread, Lock

### LOAD DATA from DB 
mutex = Lock()  
fnames = []
ffeatures = []
run = True


def load_data() :
  mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="xpto1415"
  )

  global fnames
  global ffeatures
  global run

  while run:
    _fnames = []
    _ffeatures = []

    mycursor = mydb.cursor()
    mycursor.execute("SELECT name, faceFeatures FROM facerec.faces")

    dbFaces = mycursor.fetchall()
    for x in dbFaces:
      ## print(f'face {x[0]}') 
      faceFeatures = json.loads(x[1])
      _ffeatures.append(faceFeatures)
      _fnames.append(x[0])
    
    mydb.commit()
    mycursor.close()
  ##  mydb.close()  
    

    nfaces = len(ffeatures)
    print('DB: polling')
#    print(f'faces: {nfaces}')


    mutex.acquire()
    fnames = _fnames.copy()
    ffeatures = _ffeatures.copy()
    mutex.release()
    time.sleep(2)



def process():

  global fnames
  global ffeatures
  global run
  global frame

  _fnames = []
  _ffeatures = []

  video_capture = cv2.VideoCapture(0)
  video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
  i=0
  prevnf=0;
  nf=0
  while run:
      if i % 5 == 0:
        mutex.acquire()
        _fnames = fnames.copy()
        _ffeatures = ffeatures.copy()
        mutex.release()
      i=i+1   
  
      # Grab a single frame of video
      ret, frame = video_capture.read()
      
      print(i)

      # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
      rgb_frame = frame[:, :, ::-1]

      # Find all the faces and face enqcodings in the frame of video
      face_locations = face_recognition.face_locations(rgb_frame)
      face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

      print(f'face_locations # {face_locations}')
      prevnf=nf
      nf=len(face_locations)
      print (f'prevnf: {prevnf} nf: {nf}') 
      if nf==0 and prevnf > 0:
          continue
      # Loop through each face in this frame of video

      for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          # See if the face is a match for the known face(s)
          matches = face_recognition.compare_faces(_ffeatures, face_encoding)

          name = "Unknown"

          #se the known face with the smallest distance to the new face
          face_distances = face_recognition.face_distance(_ffeatures, face_encoding)
          best_match_index = np.argmin(face_distances)
          print(np.min(face_distances))
          if matches[best_match_index] and np.min(face_distances) < 0.5 :
              name = _fnames[best_match_index]


          color = (0, 255, 0)
          if(name == "Unknown"):
              color = (0, 0, 255)
            
          # Draw a box around the face
          cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

          # Draw a label with a name below the face
          cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX
          cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

      # Display the resulting image
      cv2.imshow('Video', frame)

      # Hit 'q' on the keyboard to quit!
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

# Release handle to the webcam
  video_capture.release()
  cv2.destroyAllWindows()





def main():

    t = threading.Thread(target=load_data)
    t.start()
    process()
    global run
    run = False


if __name__ == "__main__":
    main()

