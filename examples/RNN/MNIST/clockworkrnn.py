from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.initializations import normal, identity
from einstein.layers.recurrent import SimpleRNN, ClockworkRNN
from keras.optimizers import RMSprop
from keras.utils import np_utils

import time

from einstein.fileio import FileIO

batch_size = 32
nb_classes = 10
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
X_train /= 255
X_test /= 255
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

print('Evaluate ClockworkRNN...')
model = Sequential()
model.add(ClockworkRNN(periods=[1],
                       input_dim=1,
                       output_dim=hidden_units,
                       return_sequences=False,
                       init=lambda shape: normal(shape, scale=0.001),
                       inner_init=lambda shape: identity(shape, scale=1.0),
                       activation='relu', truncate_gradient=BPTT_truncate))


model.add(Dense(hidden_units, nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test),  callbacks=[history])

scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('ClockworkRNN test score:', scores[0])
print('ClockworkRNN test accuracy:', scores[1])

record_file = FileIO()
record_file.save_pickle(history.losses, "clockworkrnn_record_titan_x_losses")
record_file.save_pickle(history.times, "clockworkrnn_record_titan_x_times")