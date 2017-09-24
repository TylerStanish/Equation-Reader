import numpy as np
np.version.version
import cv2

# import math
import os
# import cPickle
import time

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical

from sklearn.cross_validation import train_test_split

start = time.time()

root = '/Users/tylerstanish/Desktop/jsprojects/Equation-Reader/HASYv2/'

path_to_data = root + 'hasy-data-labels.csv'
data = np.genfromtxt(path_to_data, delimiter=',',usecols=np.arange(0,3), dtype=None)
files = data[:, 0]
classes = data[:, 1]
files = np.delete(files, 0)
classes = np.delete(classes, 0)

for i in range(0, files.size):
    files[i] = str(files[i]).replace('hasy-data/', '')

x = []
y = classes

num_of_classes = 0

classes_selected = []
for i in range(0, len(y)):
    if y[i] not in classes_selected:
        classes_selected.append(y[i])
        num_of_classes += 1
print(num_of_classes)
#Directory stuff
dirs = os.listdir(root+'/hasy-data')
for file in dirs:
    if file == '.DS_STORE':
        continue
    img = cv2.imread(root + '/hasy-data/' + file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, roi = cv2.threshold(img, 127, 255,cv2.THRESH_BINARY_INV)
    roi = cv2.resize(roi, (64, 64))
    x.append(roi)
    # images_arr_keys.append(f)

# cv2.imshow("new iamge", images_arr[0])
# cv2.waitKey(10000)
# cv2.destroyAllWindows()


batch_size = 32

nb_epoch = 1
nb_filters = 32
nb_pool = 2
# The dimensions of the filter
nb_conv = 3

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

uniques, id_train = np.unique(y_train, return_inverse=True)
y_train = to_categorical(id_train, num_of_classes)
uniques, id_train = np.unique(y_test, return_inverse=True)
y_test = to_categorical(id_train, num_of_classes)

# samples, channels, rows, cols
classifier = Sequential()
classifier.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(64, 64, 1)))
classifier.summary()
classifier.add(Activation('relu'))
# Maybe try making the two last arguments not a tuple
classifier.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
classifier.add(Dropout(0.5))
classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Dropout(0.5))
# I think this is where we went wrong
classifier.add(Dense(num_of_classes))
classifier.add(Activation('softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

nb_epoch = 1
batch_size = 10

x_train = np.asarray(x_train)
x_train = x_train.reshape(len(x_train), 64, 64, 1)
x_test = np.asarray(x_test)
x_test = x_test.reshape(len(x_test), 64, 64, 1)

# y_test = to_categorical(y_test)
# y_train = to_categorical(y_train)

classifier.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test))




end = time.time()
print(end-start)

