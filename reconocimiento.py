import cv2
import os
import imutils
import numpy as np

def extraccion():
    Name = 'Ana'
    Path_data = 'D:/Universidad/Semestre 1-2021/Vision Artificial/Reconocimiento Facial/Data'
    person = Path_data + '/' + Name
    if not os.path.exists(person):
        print('Carpeta creada: ',person)
        os.makedirs(person)
    
    capture = cv2.VideoCapture('Ana.mp4')
    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    while True:
            
        ret, frame = capture.read()
        if ret == False: break
        frame =  imutils.resize(frame, width=640)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(grayscale,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(person + '/rostro_{}.jpg'.format(count),rostro)
            count = count + 1
        cv2.imshow('frame',frame)
        k =  cv2.waitKey(1)
        if k == 27 or count >= 300:
            break
    capture.release()
    cv2.destroyAllWindows()
    
    
def training():
    Path_data = 'D:/Universidad/Semestre 1-2021/Vision Artificial/Reconocimiento Facial/Data'
    Listofpeople = os.listdir(Path_data)
    print('Lista de personas: ', Listofpeople)
    labels = []
    facesData = []
    label = 0
    for nameDir in Listofpeople:
        person = Path_data + '/' + nameDir
        print('Leyendo las imágenes')
        for fileName in os.listdir(person):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(person+'/'+fileName,0))
            image = cv2.imread(person+'/'+fileName,0)
            #cv2.imshow('image',image)
            #cv2.waitKey(10)
        label = label + 1
    print('labels= ',labels)
  
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))

    face_recognizer.write('modeloEigenFace.xml')
    print("Modelo listo")
    
def prueba():
    Path_data = 'D:/Universidad/Semestre 1-2021/Vision Artificial/Reconocimiento Facial/Data' #Cambia a la ruta donde hayas almacenado Data
    imagePaths = os.listdir(Path_data)
    print('imagePaths=',imagePaths)
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
 
    face_recognizer.read('modeloEigenFace.xml')
    #cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap = cv2.VideoCapture('test.mp4')
    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        ret,frame = cap.read()
        if ret == False: break
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = grayscale.copy()
        faces = faceClassif.detectMultiScale(grayscale,1.3,5)
        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            
            # EigenFaces
            if result[1] < 6300:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
       
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
#extraccion()
#training()
prueba()

    
