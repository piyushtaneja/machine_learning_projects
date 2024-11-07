import cv2  
import numpy as np  
import os

#init camera 
cap=cv2.VideoCapture(0)
#face detection using haarcascade frontalface 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='./data/'
file_name=input("enter the name of the person : ")
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

while True:
    ret,frame=cap.read()
    if ret==False:
        continue 
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])

    print(faces)
    for face in faces[-1:] :#prcoess only the larget detected face
        x,y,w,h=face 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        #extract (crop out the required face ):region of interest 
        offset=10#adds padding
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]

        face_section=cv2.resize(face_section,(100,100))
        
        
    #store every 10th face 
    skip+=1 #increment the skip counter
    if skip % 10==0:#every 10 frames add the face data to the list
        face_data.append(face_section)
        
        print(len(face_data))
    cv2.imshow("frame",frame)#display the current video frame
    cv2.imshow("face section",face_section)#display the cropped face

    key_pressed=cv2.waitKey(1)& 0xFF
    if key_pressed==ord('q'):
        break 
#convert our face list into a numpy array  
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save this data into file system 
np.save(dataset_path  + file_name + '.npy',face_data)
print("data successfully save at "+dataset_path +file_name+'.npy')
cap.release()
cv2.destroyAllWindows()      
