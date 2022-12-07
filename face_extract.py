import numpy as np
import cv2
from matplotlib import pyplot as plt
# https://minimin2.tistory.com/139
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
# https://stackoverflow.com/questions/30508922/error-215-empty-in-function-detectmultiscale

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

for i in range(1,1869):
    img = cv2.imread("./epi4_30_face/%d.jpg"%i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    flag=0
    for (x,y,w,h) in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            flag=1
            # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        if flag==1:
            if i%18==0:
                # cv2.imwrite("./epi1_100_face/%d.jpg" % (num/20), img) # 추출된 이미지가 저장되는 경로와 파일명을 지정.
                cropped=img[y :y + h , x :x + w]
                cv2.imwrite("./epi4_100_face/cropped_%d.png"%(i/18), cropped)
                flag=0
                print("i is %d"%i)
        

    # cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()