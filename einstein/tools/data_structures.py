import theano
from numpy import array

class RingBuffer:

    def __init__(self, size, ivalue = None):
        self.data = [ivalue for i in xrange(size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data

