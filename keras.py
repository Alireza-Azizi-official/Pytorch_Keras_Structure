import numpy as np 
from keras._tf_keras.keras.layers import Flatten, Conv2D, Activation, MaxPooling2D, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.optimizers import RMSprop
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.utils import to_categorical
import cv2 as cv 

# load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape and normalize data 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# convert lables into one hot encoding 
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# build model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile model 
opt = RMSprop(learning_rate = 0.0001, decay = 1e-6)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

# early stopping 
early_stopping_monitor = EarlyStopping(patience = 2)

# train model 
model.fit(x_train, y_train, batch_size = 32, epochs = 15, verbose = 1, validation_data = (x_test, y_test), callbacks = [early_stopping_monitor])

# evaluate model 
score = model.evaluate(x_test, y_test, batch_size = 32)
print(f' TEST LOSS     ==> {score[0]}')
print(f' TEST ACCURACY ==> {score[1]}')

# model summary 
model.summary()



