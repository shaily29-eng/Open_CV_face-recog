import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0) #That's the index of the camera it is used to select different cameras if you have more than one attached. By default 0 is your main one.
count = 0

while True:
    ret, frame = cap.read() #"Frame" will get the next frame in the camera (via "cap"). "Ret" will obtain return value from getting the camera frame, either true of false.
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==10:##So, if you use waitKey(0) you see a still image until you actually press something while for waitKey(1) the function will show a frame for at least 1 ms only.

        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete!!!')


#https://stackoverflow.com/questions/53731271/how-to-trigger-parameter-hints-in-visual-studio-code
#https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html