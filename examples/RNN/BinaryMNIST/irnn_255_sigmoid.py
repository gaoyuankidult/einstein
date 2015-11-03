"""
This file uses value from 1-255 as input
"""


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.initializations import normal, identity
from einstein.layers.recurrent import SimpleRNN, LSTM, SGU
from keras.optimizers import RMSprop
from keras.utils import np_utils

import time
import numpy

from einstein.fileio import FileIO

ClassModel = SimpleRNN

batch_size = 32
nb_classes = 2
nb_epochs = 200
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0
BPTT_truncate = 28*28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
#X_train /= 255
#X_test /= 255

def get_two_classes(y, label1, label2):
    assert label1 is not label2
    return numpy.logical_or([y == label1], [y == label2])[0]


X_train = X_train[get_two_classes(y_train, 0, 1)]
y_train = y_train[get_two_classes(y_train, 0, 1)]


X_test = X_test[get_two_classes(y_test, 0, 1)]
y_test = y_test[get_two_classes(y_test, 0, 1)]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.times = []
        self.start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time()-self.start_time)
        self.losses.append(logs.get("val_acc"))
history = LossHistory()

print('Evaluate %s...' % ClassModel.__class__.__name__)
model = Sequential()
model.add(ClassModel(input_dim=1, output_dim=hidden_units,
                    init=lambda shape: normal(shape, scale=0.001),
                    inner_init=lambda shape: identity(shape, scale=1.0),
                    activation='relu', truncate_gradient=BPTT_truncate))
model.add(Dense(hidden_units, nb_classes))
model.add(Activation('sigmoid'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test),  callbacks=[history])

scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('%s test score:' % ClassModel.__class__.__name__, scores[0])
print('%s test accuracy:' % ClassModel.__class__.__name__, scores[1])

record_file = FileIO()
record_file.save_pickle(history.losses, "%s_record_titan_x_losses" % ClassModel.__class__.__name__.lower())
record_file.save_pickle(history.times, "%s_record_titan_x_times" % ClassModel.__class__.__name__.lower())
