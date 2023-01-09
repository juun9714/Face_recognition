import cv2
# 영상의 이미지를 연속적으로 추출할 수 있게 하는 파이썬 라이브러리

vidcap = cv2.VideoCapture('./episode4.avi') #영상이 있는 경로, 해당 경로에 있는 파일을 이미지 추출할 대상으로 가져옴
# 4화의 파일명을 episode4로 변경 후에, 해당 파이썬 파일과 동일한 폴더에 저장했음. 

count = 0 # 추출 이미지 개수 카운트

while(vidcap.isOpened()):
    # 영상이 지속되는 동안 반복
    ret, image = vidcap.read() # 1프레임당 1장 추출, 이미지 파일은 image 변수에 저장
    image = cv2.resize(image, (960, 540)) # 이미지 사이즈 960x540으로 변경
    
    
    if(int(vidcap.get(1)) % 30 == 0): 
        # vidcap.get(1) -> 현재 프레임 숫자 반환, 프레임 숫자를 15로 나눈 나머지가 0이면 (1초당 2장, 만약 숫자가 15가 아니고 30이면 1초당 1장)
        cv2.imwrite("./epi4/15frame_960x540/%d.jpg" % count, image) # 추출된 이미지가 저장되는 경로와 파일명을 지정.
        count += 1 # 저장한 이미지 개수 카운트

    if((int(vidcap.get(1))% 1000==1)):
        print('Saved frame%d.jpg'%count) # 1000장 저장할 때마다 터미널에 알림
        
vidcap.release() # close video file