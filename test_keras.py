from tensorflow import keras
import numpy as np
import cv2
model = keras.models.load_model('model/keras_model/saves/cnn_1.h5')
img = cv2.imread('meme2.jpg')
img = cv2.resize(img, (299, 299))
img = img.reshape(-1, 299, 299, 3).astype('float32')
img = img / 255
print(img)
pred = model.predict(img, verbose=1)
print(pred)
if pred[0][0] == 1:
    print('Is meme')
else:
    print('Not meme')
