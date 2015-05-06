__author__ = 'gao'

import theano.tensor as T
from theano.tensor import shared
from abc import ABCMeta
from abc import abstractmethod


class Layer(object):
    __metaclass__ = ABCMeta
    def __init__(self, name, inputs):
        self.name = name
        self.set_inputs(inputs)
        self.build_params = {}

    def set_inputs(self, inputs):
        self.inputs = inputs

    @abstractmethod
    def build(self):
        pass


class DenseLayer(Layer):
    def __init__(self,
                 rng,
                 init_cls,
                 n_in,
                 n_out,
                 f_act,
                 inputs=None):
        super(DenseLayer, self).__init__('DenseLayer', inputs)
        self.rng = rng
        self.init_cls = init_cls
        self.n_in = n_in
        self.n_out = n_out
        self.f_act = f_act
        self.params = []
        self.__init_weights()

    def __init_weights(self):

        self.Wio = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_in,
                size_y=self.n_out,
                sparsity=-1,
            ),
            name='Wio'
        )

        self.params += [self.Wio]

        self.bo = shared(
            self.init_cls.bias_init(
                self.rng,
                size_x=self.n_out,
                sparsity=-1,
            ),
            name='bo'
        )

        self.params += [self.bo]

    def build(self):
        assert self.inputs != None, "The input of %s is None. Abort ..." % self.name
        self.outputs = self.f_act(T.dot(self.inputs, self.Wio)) + self.bo

