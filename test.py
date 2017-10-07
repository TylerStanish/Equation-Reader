import numpy as np
import cv2
import keras
import pickle

classifier = keras.models.load_model('output/model.h5')
encoder = pickle.load(open('output/encoder.p', 'rb'))

img = cv2.imread('img9.jpg', 0)
# img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret, roi = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# img = cv2.resize(img, (64, 64))

# cv2.imshow('what machine sees', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

_imagearr = []
_imagearr.append(img)
_imagearr = np.array(_imagearr)
_imagearr = _imagearr.reshape(_imagearr.shape + (1,))
predictions = classifier.predict(_imagearr, batch_size=10)
# print(predictions)

ind = np.argpartition(predictions[0], -4)[-4:]
ind[np.argsort(predictions[0][ind])]
ind = ind[::-1]
print(ind)
res = encoder.inverse_transform(ind)
print(res)
