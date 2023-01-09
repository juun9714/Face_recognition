import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

for i in range(1,5000):
    img = cv2.imread("./night/epi4_30_face/%d.jpg"%i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    flag=0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            flag=1
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        
        if flag==1:
            cv2.imwrite("./epi1_100_face/%d.jpg" % i, img) # 추출된 이미지가 저장되는 경로와 파일명을 지정.
            flag=0
        

    # cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()