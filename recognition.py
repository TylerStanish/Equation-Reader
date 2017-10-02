import cv2
import math
import keras
import pickle
import json
import numpy as np

from operands import Operand
from operands import Equation

# This is for testing purposes only
encoder = pickle.load(open('encoder.p', 'rb'))
# model = keras.models.load_model('output/model.h5')
with open('model.json') as data_file:
    data = json.load(data_file)
model = keras.models.model_from_json(data)

# Function declarations
def get_coords_of_adjacent(ncp, ind):
    x0 = y0 = h0 = w0 = 0
    x1 = y1 = h1 = w1 = 0
    x, y, w, h = cv2.boundingRect(ncp[ind])
    if ind == 0:
        x1, y1, w1, h1, = cv2.boundingRect(ncp[ind + 1])
    elif ind == len(ncp) - 1:
        x0, y0, w0, h0, = cv2.boundingRect(ncp[ind - 1])
    else:
        x0, y0, w0, h0, = cv2.boundingRect(ncp[ind - 1])
        x1, y1, w1, h1, = cv2.boundingRect(ncp[ind + 1])
    return x0, y0, h0, w0, x, y, w, h, x1, y1, w1, h1

# def check_for_fraction(image, equation, ncp, ind):


def check_for_arrows(image, equation, ncp, ind):
    x0, y0, h0, w0, x, y, w, h, x1, y1, w1, h1 = get_coords_of_adjacent(ncp, ind)

    if math.fabs(x - x0) < 0.4 * w and math.fabs((x0 + w0) - (x + w)) < 0.4 * w and y0 < y:
        obj = Operand(
            x=x,
            y=y,
            top=Operand(x=x0, y=y0, image=image[y0:y0 + h0, x0:x0 + w0]),
            image=image[y:y + h, x:x + w]
        )
        equation.add_operand(obj)
        return True
    elif math.fabs(x - x1) < 0.4 * w and math.fabs((x1 + w1) - (x + w)) < 0.4 * w and y1 < y:
        obj = Operand(
            x=x,
            y=y,
            top=Operand(x=x1, y=y1, image=image[y1:y1 + h0, x1:x1 + w1]),
            image=image[y:y + h, x:x + w]
        )
        equation.add_operand(obj)
        return True
    else:
        equation.add_operand(Operand(x=x, y=y, image=image[y:y + h, x:x + w]))
        return False

def resize_to_classify(img):
    roi = cv2.resize(img, (64, 64))
    _imagearr = []
    _imagearr.append(roi)
    _imagearr = np.array(_imagearr)
    # _imagearr = np.expand_dims(_imagearr, axis=0)
    return _imagearr.reshape(_imagearr.shape + (1,))


def read_image(uri):
    image = cv2.imread(uri)
    # image = cv2.resize(image, (700, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray)
    # cv2.imshow('de-noised', gray)
    # cv2.waitKey(0)
    edged = cv2.Canny(gray, 30, 130)
    # cv2.imshow('Canny edges', edged)
    # cv2.waitKey(0)
    hierarchy, contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Now to sort out the contours
    new_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 5 or w < 5:
            continue
        new_contours.append((x, gray[y:y+h, x:x+w]))
    new_contours = sorted(new_contours, key=lambda x: x[0])
    nc = [a[1] for a in new_contours]
    nc = [resize_to_classify(a) for a in nc]

    equation = Equation(image=image)
    i = 0
    selected = []

    for c in nc:
        if c in selected:
            continue
        # first classify fractions
        classification = classify(c)
        print(classification)
        if classification == '-':
            print('classify as fraction')
        # selected.append(Operand())


def classify(contour):
    predictions = model.predict(contour)
    return encoder.inverse_transform(predictions)

read_image('test_images/img_1330.JPG')
cv2.destroyAllWindows()
