__author__ = 'gao'



from keras.initializations import *

def identity(shape):
    return shared_identity(shape[0])
from einstein.tools.theano_utils import shared_identity


