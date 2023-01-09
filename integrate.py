import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


for i in range(6,25):
    vidcap = cv2.VideoCapture('./theKingOfAmbition/episode%d.avi' % i)
    count = 1 # 추출 이미지 개수 카운트
    real_count=1
    os.mkdir("./night/epi%d_30_face_960x540"%i)  # 폴더 생성
    while(vidcap.isOpened()):
        if real_count>100:
            vidcap.release()
            break
        # 영상이 지속되는 동안 반복
        ret, image = vidcap.read()
        image = cv2.resize(image, (960, 540)) # 이미지 사이즈 960x540으로 변경
        
        if(int(vidcap.get(1)) % 30 == 0): 
            # vidcap.get(1) -> 현재 프레임 숫자 반환, 프레임 숫자를 15로 나눈 나머지가 0이면 (1초당 2장, 만약 숫자가 15가 아니고 30이면 1초당 1장)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            flag=0

            for (x,y,w,h) in faces:
                # cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex,ey,ew,eh) in eyes:
                    flag=1
                    # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


                if flag==1:
                    if count%15==0:
                        cropped=image[y :y + h , x :x + w]
                        # cv2.imwrite("./night/100/epi4_100_face/cropped_%d.png"%(i/18), cropped)
                        cv2.imwrite("./night/epi{}_30_face_960x540/cropped_{}.jpg".format(i, real_count),cropped) # 추출된 이미지가 저장되는 경로와 파일명을 지정.
                        real_count+=1
                        print("Saved %d'th face cropped image"%real_count)
                    count += 1 # 저장한 이미지 개수 카운트
                    flag=0
            
    vidcap.release() # close video file

# frame extract done, face extract starts