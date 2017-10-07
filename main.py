import numpy as np
np.version.version
import cv2

# import math
import os
# import cPickle
import time
import pickle
import keras

from keras.models import Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical

from sklearn.cross_validation import train_test_split

start = time.time()

root = '/data'
# root = 'extracted_images'

x = []
y = []

dirs = os.listdir(root)
for dir in dirs:
    if dir[:1] == '.':
        continue
    if dir == 'ldots' or dir == ',' or dir == '.' or dir == 'times' or dir == '!':
        continue
    imgs = os.listdir(root + '/' + dir)
    print('scanning ' + dir)
    for ind, im in enumerate(imgs):
        if im[:1] == '.':
            continue
        # Please remove this for production
        if ind > 1000:
            continue

        img = cv2.imread(root + '/' + dir + '/' + im, 0)
        # ret, roi = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        img = cv2.resize(img, (64, 64))
        # img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
        _, contours, __ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        white = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        white[:] = (0)
        white = cv2.drawContours(white, contours, -1, (255, 255, 255), thickness=10)
        # for c in contours:
        #     x1, y1, w, h = cv2.boundingRect(c)
        #     if h < 4 or w < 4:
        #         continue
        #
        #
        #
        #     white = white[y1:y1 + h, x1:x1 + w]

            # kernel = np.ones((5, 5), np.uint8)
            # white = cv2.dilate(white, kernel=kernel, iterations=1)
            # white = cv2.resize(white, (128, 128))
            # white = cv2.copyMakeBorder(white, 3, 3, 3, 3, cv2.BORDER_REPLICATE, value=[255, 255, 255])
            # for pixel in white:
            #     print(pixel)
            # cv2.imshow('reg', img)
            # cv2.waitKey(0)

        white = cv2.resize(white, (64, 64))
        x.append(white)
        # cv2.imshow('white', white)
        # cv2.waitKey(0)
        y.append(dir)

num_of_classes = 0

classes_selected = []
for i in range(0, len(y)):
    if y[i] not in classes_selected:
        classes_selected.append(y[i])
        num_of_classes += 1

batch_size = 32

nb_epoch = 1
nb_filters = 32
nb_pool = 2
# The dimensions of the filter
nb_conv = 3

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_test = label_encoder.fit_transform(y_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# uniques, id_train = np.unique(y_train, return_inverse=True)
# y_train = to_categorical(id_train, num_classes=num_of_classes)
# uniques, id_test = np.unique(y_test, return_inverse=True)
# y_test = to_categorical(id_test, num_classes=num_of_classes)

# samples, channels, rows, cols
classifier = Sequential()
classifier.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(64, 64, 1)))
# classifier.summary()
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
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

nb_epoch = 10
batch_size = 10

x_train = np.asarray(x_train)
x_train = x_train.reshape(len(x_train), 64, 64, 1)
x_test = np.asarray(x_test)
x_test = x_test.reshape(len(x_test), 64, 64, 1)

# y_test = to_categorical(y_test)
# y_train = to_categorical(y_train)

classifier.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_data=(x_test, y_test))

end = time.time()
print(end-start)


classifier.save('/output/model.h5')
pickle.dump(encoder, open('/output/encoder.p', 'wb'), protocol=2)


