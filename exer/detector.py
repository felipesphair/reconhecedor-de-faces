# IMPORTS
import numpy as np
import cv2
import os



def savePerson():
    global ultimoNome
    global boolSaveimg
    print('Qual é o seu nome?')
    name = input()
    ultimoNome = name
    boolSaveimg = True
    
def trainData():
    global recognizer
    global trained
    global persons
    trained = True
    persons = os.listdir('train')

    ids = []

    faces = []

    for i, p in enumerate(persons):
        for f in os.listdir(f'train/{p}'):
            img = cv2.imread(f'train/{p}/{f}',0)
            faces.append(img)
            ids.append(i)
    recognizer.train(faces, np.array(ids))

def saveImg(img):
    global ultimoNome

    if not os.path.exists('train'):
        os.makedirs('train')

    if not os.path.exists(f'train/{ultimoNome}'):
        os.makedirs(f'train/{ultimoNome}')

    files = os.listdir(f'train/{ultimoNome}')
    cv2.imwrite(f'train/{ultimoNome}/{str(len(files))}.jpg', img)


ultimoNome = ''
boolSaveimg = False

trained = False

savecount = 0

persons = []

recognizer = cv2.face.LBPHFaceRecognizer_create() 

# READ VIDEO
cap = cv2.VideoCapture(0)

# LOAD HAAR CASCADE CLASSIFIER
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# LOOP
while(True):
    
    # READ FRAME
    _, frame = cap.read()

    # GRAY FRAME
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # DETECT FACES IN FRAME
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # RUN ALL FACES IN FRAME
    for (x,y,w,h) in faces:

        roi = gray[y:y+h, x:x+w]

        roi = cv2.resize(roi, (50,50))

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        if trained:
            idf, conf = recognizer.predict(roi)
            nameP = persons[idf]
            cv2.putText(frame, nameP, (x, y), 1, 1, (0,255,0), 1, cv2.LINE_AA)
        
        if boolSaveimg:
            saveImg(roi)
            savecount += 1
            

        if savecount > 50: 
            boolSaveimg = False
            savecount = 0

    cv2.imshow('frame',frame)

    key = cv2.waitKey(1)
    # WAITKEY
    if key == ord('s'):
        savePerson()

    if key == ord('t'):
        trainData()

    if key == ord('q'):
        break

# RELEASE CAP
cap.release()

# DESTROY ALL WINDOWS
cv2.destroyAllWindows()