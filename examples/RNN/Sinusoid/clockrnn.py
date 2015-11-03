from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from einstein.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from einstein.layers.recurrent import LSTM, StackableGRU, StackableSGU, SGUModified1, DSGU, SGU, ClockworkGRU, ClockworkSGU
from keras.datasets import imdb

import einstein as E
maxlen = 100
print("Loading data...")
TAU = 2 * np.pi
BATCH_SIZE = 2
T = np.linspace(0, TAU, 256)
COEFFS = ((2, 1.5), (3, 1.8), (4, 1.1),(0.1, 0.2))
SIN = sum(c * np.sin(TAU * f * T) for c, f in COEFFS)
WAVES = np.concatenate([SIN[:, None, None]] * BATCH_SIZE, axis=1).astype('f')
ZERO = np.zeros((len(T), BATCH_SIZE, 1), 'f')
ax = plt.subplot(111)
ax.plot(T, SIN, ':', label='Target', alpha=0.7)

print(ZERO, WAVES)

(X_train, y_train) = (ZERO, WAVES)

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

print('X_train shape:', X_train.shape)




