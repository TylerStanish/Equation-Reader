import numpy as np
import cv2
import keras
import pickle

classifier = keras.models.load_model('model.h5')
encoder = pickle.load(open('encoder.p', 'rb'))

img = cv2.imread('test_images/img_15.jpg', 0)
# img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret, roi = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
roi = cv2.resize(img, (64, 64))

cv2.imshow('what machine sees', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

_imagearr = []
_imagearr.append(roi)
_imagearr = np.array(_imagearr)
_imagearr = _imagearr.reshape(_imagearr.shape + (1,))
predictions = classifier.predict(_imagearr)

res = encoder.inverse_transform(predictions)
print(res)
