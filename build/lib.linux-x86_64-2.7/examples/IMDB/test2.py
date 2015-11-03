import numpy
import theano
import theano.tensor as T

theano.config.compute_test_value = 'off'

W1val = numpy.random.rand(1, 128).astype(theano.config.floatX)

W1 = theano.shared(W1val, 'W1')

x  = T.matrix('x')

func_of_W1 = W1

h1 = x * func_of_W1

f = theano.function([x], h1)

print f(numpy.random.rand(1, 128).astype(theano.config.floatX)).shape