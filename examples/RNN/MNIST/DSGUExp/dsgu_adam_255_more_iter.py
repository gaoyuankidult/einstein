"""
This file uses value from 1-255 as input
"""


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

import keras
#import os
#import gzip
#import sys
#import six
#from urllib import urlretrieve

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.initializations import normal, identity
from einstein.layers.recurrent import SimpleRNN, LSTM, SGU, DSGU
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras.datasets import mnist

import time
import numpy

from einstein.fileio import FileIO

def get_file(fname, origin, untar=False):
    datadir = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    try:
        f = open(fpath)
    except:
        print('Downloading data from',  origin)

        global progbar
        progbar = None
        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count*block_size)

        urlretrieve(origin, fpath, dl_progress)
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            tfile.extractall(path=datadir)
            tfile.close()
        return untar_fpath

    return fpath


def load_data(path="mnist.pkl.gz"):
    path = get_file(path, origin="https://s3.amazonaws.com/img-datasets/mnist.pkl.gz")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = six.moves.cPickle.load(f)
    else:
        data = six.moves.cPickle.load(f, encoding="bytes")
    print(data)
    f.close()

    return data # (X_train, y_train), (X_test, y_test)


run_time = 10
ClassModel = DSGU




batch_size = 32
nb_classes = 10
nb_epochs = 200
hidden_units = 100

learning_rate = 1e-3
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

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

for i in xrange(run_time):
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.val_losses = []
            self.times = []
            self.start_time = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time()-self.start_time)
            self.val_losses.append(logs.get("val_acc"))
    history = LossHistory()

    print('Evaluate %s...' % ClassModel.__class__.__name__)
    model = Sequential()
    model.add(ClassModel(1, hidden_units, ))
    model.add(Dense(hidden_units, nb_classes))
    model.add(Activation('sigmoid'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test),  callbacks=[history])

    scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('%s test score:' % ClassModel.__class__.__name__, scores[0])
    print('%s test accuracy:' % ClassModel.__class__.__name__, scores[1])

    record_file = FileIO()
    record_file.save_pickle(history.val_losses, "dsgu_adam_%s_%d_255_more_iter_titan_x_losses" % (ClassModel.__class__.__name__.lower(), i))
    record_file.save_pickle(history.times, "dsgu_adam_%s_%d_255_more_iter_titan_x_times" % (ClassModel.__class__.__name__.lower(), i))
