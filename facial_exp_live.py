'''
Github Id: shantanugupta1118
Programmer: Shantanu Gupta (PSIT B.Tech - CSE, (Second year))

'''
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier= load_model('Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']


#camera
cap = cv2.VideoCapture(0)


while True:
    #Grab a single frame of video
    ret, frame = cap.read()
    labels= []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),1)
        # rectangle( frame, frame size, frame dimensions, color of frame(Blue, green, reg)BGR,
                    #size of strip of frame)
                    
        roi_gray = gray[y:y+h, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
                                    #   48,48 mean as resize upto 48 x x48
        #roi_gray manipulate for face detector 
        #rext, face, image = face_detector(frame)
        
        
        ''' make condition 
                    if sum of roi_gray is not equal to 0
                    or 
                    if roi_gray detect any face in a frame then condition will true else 
                    false.
                    '''
        if np.sum([roi_gray]) != 0:     
            roi = roi_gray.astype('float')/255.0       #value of roi_gray divided by 255 because any value of 
                                                            #image will not greater than 255
                                                            
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
        #make a prediction on the ROI, then lookup the class
            
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
            
            #text color will be green and text size will 3px
                                             
            
        else:
            cv2.putText(frame,'No Face Found',(20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
            #if face not found then NO FACE FOUND will occur
            
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # when in screen press q key then it will stop
    
cap.release()
cv2.destroyAllWindowa()
            
