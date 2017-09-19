from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import cgi
import cgitb

classifier = Sequential()

# Because I think you're using theno as the backend you have to enter the coordinates
# as (3, 64, 64)
classifier.add(Convolution2D(32, 3, 3, input_shape=(3, 64, 64), activation='relu'))
print('bla');

