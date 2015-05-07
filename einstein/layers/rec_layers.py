__author__ = 'gao'

from theano import config
from theano import shared
import theano.tensor as T
from theano import scan
from theano import function
from theano import grad
from numpy import arange
from numpy import zeros
from basic_layers import Layer
from basic_layers import DenseLayer

config.mode = 'FAST_COMPILE'



class LSTMLayer(Layer):
    def __init__(self,
                 rng,
                 init_cls,
                 n_steps,
                 n_in,
                 n_units,
                 inputs=None
                 ):
        super(LSTMLayer, self).__init__('LSTM', inputs)
        # hidden, cell, gates should be of same size?
        self.rng = rng
        self.init_cls = init_cls
        self.n_steps = n_steps
        self.params = []
        self.n_hidden = n_units
        self.n_in = n_in
        self.n_ig = n_units
        self.n_c = n_units
        self.n_fg = n_units
        self.n_og = n_units
        self.__init_weighs()
        self.build_params['n_steps'] = n_steps

    def __init_weighs(self):
        self.Wxi = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_in,
                size_y=self.n_ig,
                sparsity=-1,
            ),
            name="Wxi"
        )
        self.params += [self.Wxi]

        self.Wci = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_c,
                size_y=self.n_ig,
                sparsity=-1,
            ),
            name="Wci"
        )

        self.params += [self.Wci]

        self.Whi = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_hidden,
                size_y=self.n_ig,
                sparsity=-1,
            ),
            name="Whi"
        )
        self.params += [self.Whi]

        self.Wxf = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_in,
                size_y=self.n_fg,
                sparsity=-1,
            ),
            name="Wxf"
        )
        self.params += [self.Wxf]

        self.Wcf = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_c,
                size_y=self.n_fg,
                sparsity=-1,
            ),
            name="Wcf"
        )

        self.params += [self.Wcf]

        self.Whf = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_hidden,
                size_y=self.n_fg,
                sparsity=-1,
            ),
            name="Whf"
        )

        self.params += [self.Whf]

        self.Wxc = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_in,
                size_y=self.n_c,
                sparsity=-1,
            ),
            name="Wxc"
        )

        self.params += [self.Wxc]

        self.Whc = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_hidden,
                size_y=self.n_c,
                sparsity=-1,
            ),
            name="Whc"
        )

        self.params += [self.Whc]

        self.Wxo = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_in,
                size_y=self.n_og,
                sparsity=-1,
            ),
            name="Wxo"
        )

        self.params += [self.Wxo]

        self.Who = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_hidden,
                size_y=self.n_og,
                sparsity=-1,
            ),
            name="Who"
        )

        self.params +=[self.Who]

        self.Wco = shared(
            self.init_cls.weight_init(
                self.rng,
                size_x=self.n_c,
                size_y=self.n_og,
                sparsity=-1,
            ),
            name="Wco"
        )

        self.params +=[self.Wco]

        self.bi = shared(
            self.init_cls.bias_init(
                self.rng,
                size_x=self.n_ig,
                sparsity=-1,
            ),
            name="bi"
        )

        self.params += [self.bi]

        self.bf = shared(
            self.init_cls.bias_init(
                self.rng,
                size_x=self.n_fg,
                sparsity=-1,
            ),
            name="bf"
        )

        self.params += [self.bf]

        self.bc = shared(
            self.init_cls.bias_init(self.rng,
                size_x=self.n_c,
                sparsity=-1,
            ),
            name="bc"
        )

        self.params += [self.bc]

        self.bo = shared(
            self.init_cls.bias_init(self.rng,
                size_x=self.n_og,
                sparsity=-1,
            ),
            name="bo"
        )

        self.params += [self.bo]

    def build(self,
              n_steps,
              learning_rate=1e-3,
              momentum=0.6,
              weight_decay=0.01,
              ):

        def __step_fprop(u_t, c_tm1, h_tm1,
                        Wxi, Whi, Wci, bi,
                        Wxf, Whf, Wcf, bf,
                        Wxc, Whc, bc,
                        Wxo, Who, Wco, bo,
                        ):
            # input gate
            ig = T.nnet.sigmoid(T.dot(u_t, Wxi) +
                                T.dot(h_tm1, Whi) +
                                T.dot(c_tm1, Wci) +
                                bi)
            # forget gate
            fg = T.nnet.sigmoid(T.dot(u_t, Wxf) +
                                T.dot(h_tm1, Whf) +
                                T.dot(c_tm1, Wcf) +
                                bf)

            # cell
            cc= fg * c_tm1 + ig * T.tanh(T.dot(u_t, Wxc) +
                                       T.dot(h_tm1, Whc) +
                                       bc)
            #  output gate
            og = T.nnet.sigmoid(T.dot(u_t, Wxo) +
                                T.dot(h_tm1,Who)  +
                                T.dot(c_tm1, Wco) +
                                bo)
            # hidden state
            hh = og * T.tanh(cc)

            return cc, hh

        # initial hidden state of the RNN
        c0 = shared(zeros([self.n_c, 1], dtype=config.floatX))
        # initial hidden state of the RNN
        h0 = shared(zeros([self.n_hidden, 1], dtype=config.floatX))

        [_, hh], updates = scan(fn=__step_fprop,
                           sequences=self.inputs,
                           outputs_info=[c0, h0],
                           non_sequences=[self.Wxi, self.Whi, self.Wci, self.bi,
                                          self.Wxf, self.Whf, self.Wcf, self.bf,
                                          self.Wxc, self.Whc, self.bc,
                                          self.Wxo, self.Who, self.Wco, self.bo
                                          ],
                           n_steps=n_steps
                           )
        self.outputs = hh

        # layer_cost = T.sum((targets - predicts)**2)
        # grads = [grad(layer_cost, param) for param in self.params]
        # updates = []
        #
        # for param_i, grad_i in zip(self.params, grads):
        #     mparam_i = shared(zeros(param_i.get_value().shape, dtype=config.floatX))
        #     full_grad = grad_i + weight_decay * param_i
        #     v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        #     w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        #     updates.append((mparam_i, v))
        #     updates.append((param_i, w))
        #
        # self.__train = function(inputs=[model_inputs, targets],
        #                  outputs=layer_cost,
        #                  updates=updates,)
        #
        # self.__predicts = function(inputs=[model_inputs],
        #                   outputs=predicts)




if __name__ == "__main__":
    pass
    # from numpy.random import RandomState
    # from initializer.rand_init import RandomSparseInit
    # from databank import sin_gene
    #
    # rng = RandomState()
    # init_cls = RandomSparseInit()
    # lstmlayer = LSTMLayer(rng=rng,
    #                       init_cls=init_cls,
    #                       n_steps=10,
    #                       n_units=10,
    #                       n_in=1,
    #                       )
    # lstmlayer.train(epochs=100)





