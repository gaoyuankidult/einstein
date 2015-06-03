import numpy as np

from keras.utils.theano_utils import *

def shared_identity(dim, dtype=theano.config.floatX, name=None):
    return sharedX(np.identity(dim), dtype=dtype, name=name)