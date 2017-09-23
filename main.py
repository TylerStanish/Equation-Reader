import numpy as np
np.version.version
import cv2

# import math
import os
# import cPickle
import time

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# classifier = Sequential()


start = time.time()

root = '/Users/tylerstanish/Desktop/jsprojects/Equation-Reader/HASYv2/'

path_to_data = root + 'hasy-data-labels.csv'
data = np.genfromtxt(path_to_data, delimiter=',',usecols=np.arange(0,3), dtype=None)
keys = data[:, 0]
values = data[:, 1]
keys = np.delete(keys, 0)
values = np.delete(values, 0)
for i in range(0, keys.size):
    keys[i] = str(keys[i]).replace('hasy-data/', '')

images_arr = []
# images_arr_keys = []

#Directory stuff
dirs = os.listdir(root+'/hasy-data')
for file in dirs:
    if file == '.DS_STORE':
        continue
    img = cv2.imread(root + '/hasy-data/' + file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, roi = cv2.threshold(img, 127, 255,cv2.THRESH_BINARY_INV)
    images_arr.append(roi)
    # images_arr_keys.append(f)

# cv2.imshow("new iamge", images_arr[0])
# cv2.waitKey(10)
# cv2.destroyAllWindows()


end = time.time()
print(end-start)

