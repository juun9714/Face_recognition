from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
'''
optimizer -> https://89douner.tistory.com/274
rmsprop -> https://keras.io/api/optimizers/rmsprop/
pyplot -> https://codetorial.net/tensorflow/visualize_training_history.html
'''

# 카테고리 지정하기
categories = ['happy', 'embarrass', 'wrath', 'anxious', 'hurt', 'sad', 'neutral']
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64
image_h = 64
# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("./7obj.npy", allow_pickle=True)
# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)
# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer=optimizers.RMSprop(learning_rate=0.001),
    metrics=["acc"])
# 모델 확인
#print(model.summary())

hdf5_file = "./7obj-model.hdf5"

history=model.fit(X_train, y_train, validation_split=0.25, batch_size=32, epochs=100)
model.save_weights(hdf5_file)
# 학습 완료된 모델 저장
# if os.path.exists(hdf5_file):
#     # 기존에 학습된 모델 불러들이기
#     model.load_weights(hdf5_file)
# else:
#     # 학습한 모델이 없으면 파일로 저장
#     history=model.fit(X_train, y_train, batch_size=32, epochs=10)
#     model.save_weights(hdf5_file)

score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc

accu=history.history['acc']
val_accu=history.history['val_acc']
plt.plot(accu, label="accuracy")
plt.plot(val_accu, label="val_accuracy")
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'],label="val_loss")

plt.xlabel('Epoch')
plt.ylabel('Acc/Loss')
plt.legend()
plt.show()
