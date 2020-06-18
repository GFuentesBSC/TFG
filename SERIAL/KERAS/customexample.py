import os
import math
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ResNet50
import time

batch_size = 32
num_classes = 10
# epochs = 100
tf.random.set_seed(1)
## simple test:
epochs = 8

((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

class MyModel(Sequential):

  def __init__(self):
    Sequential.__init__(self)
    self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    self.add(MaxPooling2D((2, 2)))
    self.add(Dropout(0.2))
    self.add(Flatten())

  def add_dense_layer(self, size):
    self.add(Dense(size, activation='relu'))

model = MyModel()
model.add_dense_layer(num_classes)
opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=.9,nesterov=False)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
