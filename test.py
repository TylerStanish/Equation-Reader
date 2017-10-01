import cv2
import pickle
import keras
import numpy as np
import json

# image = cv2.imread('img_1330.jpg')
# # image = cv2.resize(image, (700, 500))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.fastNlMeansDenoising(gray)
# cv2.imshow('de-noised', gray)
# cv2.waitKey(0)
# edged = cv2.Canny(gray, 30, 130)
# cv2.imshow('Canny edges', edged)
# cv2.waitKey(0)
# hierarchy, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# for i, c in enumerate(contours):
#     x, y, w, h = cv2.boundingRect(c)
#     if w < 4:
#         continue
#     cropped = image[y:y+h, x:x+w]
#     # cv2.imwrite('img_'+str(i)+'.jpg', cropped)
#
# cv2.destroyAllWindows()

encoder = pickle.load(open('encoder.p', 'rb'))
# model = keras.models.load_model('output/model.h5')
with open('model.json') as data_file:
    data = json.load(data_file)
model = keras.models.model_from_json(data)
img = cv2.imread('img_11.jpg', 0)
# ret, roi = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
roi = cv2.resize(img, (64, 64))
_imagearr = []
_imagearr.append(roi)
_imagearr = np.array(_imagearr)
# _imagearr = np.expand_dims(_imagearr, axis=0)
_imagearr = _imagearr.reshape(_imagearr.shape + (1,))
predictions = model.predict(_imagearr)
res = encoder.inverse_transform(predictions)
print(res)

cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
