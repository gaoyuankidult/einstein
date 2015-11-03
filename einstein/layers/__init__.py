__author__ = 'gao'

from lasagne import nonlinearities
<<<<<<< HEAD
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from recurrent import LSTM, GRU, SimpleDeepRNN, SimpleDeepGRU, PrimeDeepGRU, DoubleGatedDeepGRU, DeepGRU, SGU, ClockworkRNN, SimpleRNN, ClockworkGRU, StackableGRU, StackableSGU

from convolutional import Convolution2D

from misc import set_network_params

import embedding
=======
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import LSTMLayer
from lasagne.layers.shape import ReshapeLayer

from lasagne.layers import get_all_params

from misc import set_network_params
>>>>>>> 21c5a6a5b15be02716243cb1dc1065a0e8ee4ce2
