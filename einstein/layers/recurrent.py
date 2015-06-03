from __future__ import print_function
from keras.layers.recurrent import LSTM, GRU, SimpleDeepRNN, Layer ,SimpleRNN
from keras import activations, initializations
from einstein.layers.initializations import identity
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
import theano.tensor as TT
import theano

import einstein as E

class SimpleDeepGRU(GRU):

    def __init__(self, *args, **kwargs):

        super(SimpleDeepGRU, self).__init__(*args, **kwargs)

    def _step(self,
        xz_t, xr_t, xh_t,
        h_tm1, h_tm2,
        u_z, u_r, u_h):
        z = self.inner_activation(xz_t + TT.dot(h_tm1, u_z) + TT.dot(h_tm2, u_z))
        r = self.inner_activation(xr_t + TT.dot(h_tm1, u_r) + TT.dot(h_tm2, u_r))
        hh_t = self.activation(xh_t + TT.dot(r * h_tm1, u_h) + TT.dot(r * h_tm2, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        x_z = TT.dot(X, self.W_z) + self.b_z
        x_r = TT.dot(X, self.W_r) + self.b_r
        x_h = TT.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h],
            outputs_info=[dict(initial=E.tools.theano_utils.alloc_zeros_matrix(2, X.shape[1],  self.output_dim), taps=[-1, -2])],
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

class StackableGRU(GRU):
    def __init__(self, *args, **kwargs):
        super(StackableGRU, self).__init__(*args, **kwargs)
    def get_output(self, train):
        X = self.get_input(train)
        if X.ndim == 3:
            X = X.dimshuffle((1,0,2))

        x_z = TT.dot(X, self.W_z) + self.b_z
        x_r = TT.dot(X, self.W_r) + self.b_r
        x_h = TT.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h],
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

class DeepGRU(GRU):

    def __init__(self, depth=2, *args, **kwargs):
        super(DeepGRU, self).__init__(*args, **kwargs)
        self.depth = depth

    def _step(self, *args):
        xz_t = args[0]
        xr_t = args[1]
        xh_t = args[2]

        u_z = args[-3]
        u_r = args[-2]
        u_h = args[-1]

        z = xz_t
        r = xr_t

        for i in range(3, 3 + self.depth):
            z += TT.dot(args[i], u_z)
            r += TT.dot(args[i], u_r)

        z = self.inner_activation(z)
        r = self.inner_activation(r)


        hh_t = xh_t
        for i in range(3, 3 + self.depth):
            hh_t += TT.dot(r * args[i], u_h)
        hh_t = self.activation(hh_t)


        h_sum = args[3]
        for i in range(3+1, 3 + self.depth):
            h_sum += args[i]

        h_t = z * h_sum/self.depth + (1 - z) * hh_t
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        x_z = TT.dot(X, self.W_z) + self.b_z
        x_r = TT.dot(X, self.W_r) + self.b_r
        x_h = TT.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h],
            outputs_info=[dict(
                initial=E.tools.theano_utils.alloc_zeros_matrix(self.depth, X.shape[1],  self.output_dim),
                taps=[(-i-1) for i in range(self.depth)])],
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]


class PrimeDeepGRU(GRU):

    def __init__(self, *args, **kwargs):
        super(PrimeDeepGRU, self).__init__(*args, **kwargs)

    def _step(self,
        xz_t, xr_t, xh_t,
        h_tm1, h_tm2, h_tm3, h_tm5, h_tm7,
        u_z, u_r, u_h):
        z = self.inner_activation(xz_t + TT.dot(h_tm1, u_z) + TT.dot(h_tm2, u_z) + TT.dot(h_tm3, u_z) + TT.dot(h_tm5, u_z) + TT.dot(h_tm7, u_z))
        r = self.inner_activation(xr_t + TT.dot(h_tm1, u_r) + TT.dot(h_tm2, u_r) + TT.dot(h_tm3, u_r) + TT.dot(h_tm5, u_r) + TT.dot(h_tm7, u_r))
        hh_t = self.activation(xh_t + TT.dot(r * h_tm1, u_h) + TT.dot(r * h_tm2, u_h)  + TT.dot(r * h_tm3, u_h) + TT.dot(r * h_tm5, u_h) + TT.dot(r * h_tm7, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        x_z = TT.dot(X, self.W_z) + self.b_z
        x_r = TT.dot(X, self.W_r) + self.b_r
        x_h = TT.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h],
            outputs_info=[dict(initial=E.tools.theano_utils.alloc_zeros_matrix(5, X.shape[1],  self.output_dim), taps=[-1, -2, -3, -5, -7])],
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

class DoubleGatedDeepGRU(GRU):

    def __init__(self, *args, **kwargs):

        super(DoubleGatedDeepGRU, self).__init__(*args, **kwargs)
        self.U_z2 = self.inner_init((self.output_dim, self.output_dim))
        self.U_r2 = self.inner_init((self.output_dim, self.output_dim))
        self.U_h2 = self.inner_init((self.output_dim, self.output_dim))

        self.params.extend([self.U_z2, self.U_r2, self.U_h2])

    def _step(self,
        xz_t, xr_t, xh_t,
        h_tm1, h_tm2,
        u_z, u_r, u_h, u_z2, u_r2, u_h2):
        z = self.inner_activation(xz_t + TT.dot(h_tm1, u_z))
        r = self.inner_activation(xr_t + TT.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + TT.dot(r * h_tm1, u_h))

        z2 = self.inner_activation(xz_t + TT.dot(h_tm2, u_z2))
        r2 = self.inner_activation(xr_t + TT.dot(h_tm2, u_r2))
        hh_t2 = self.activation(xh_t + TT.dot(r2 * h_tm2, u_h2))

        h_t = 1/3. * (z * h_tm1 + (1 - z) * hh_t) + 2/3. * (z2 * h_tm2 + (1 - z2) * hh_t2)
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        x_z = TT.dot(X, self.W_z) + self.b_z
        x_r = TT.dot(X, self.W_r) + self.b_r
        x_h = TT.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h],
            outputs_info=[dict(initial=E.tools.theano_utils.alloc_zeros_matrix(2, X.shape[1],  self.output_dim), taps=[-1, -2])],
            non_sequences=[self.U_z, self.U_r, self.U_h, self.U_z2, self.U_r2, self.U_h2],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]



class SimpleGatedUnit1(Layer):

    def __init__(self, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='softplus', inner_activation='hard_sigmoid',
#        gate_activation= 'relu',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(SimpleGatedUnit1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = TT.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.W_in = self.init((self.input_dim, self.output_dim))
        self.b_in = shared_zeros((self.output_dim))

        self.W_out = self.init((self.input_dim, self.output_dim))
        self.b_out = shared_zeros((self.output_dim))

        self.U_in = self.inner_init((self.output_dim, self.output_dim))
        self.bu_in = shared_zeros((self.output_dim))

        self.U_out = self.inner_init((self.output_dim, self.output_dim))
        self.bu_out = shared_zeros((self.output_dim))

        self.params = [
            self.W, self.U, self.b,
            self.W_out, self.b_out,
            self.W_in, self.b_in,
            self.U_in, self.U_out,
            self.bu_in, self.bu_out
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
        xx, xx_tm1,
        x_in, x_intm1,
        x_out, x_outtm1,
        h_tm1, h_tm2,
        u, u_in, u_out,
        bu_in, bu_out):
        z = self.inner_activation(xx + TT.dot(h_tm1, u) + xx_tm1)

        z_in = self.inner_activation(x_in + TT.dot(h_tm1, u_in) + x_intm1 + bu_in)
        z_out = self.activation(x_out + TT.dot(z_in * h_tm1, u_out) + x_outtm1 + bu_out)

        h_t = (1 - z) * h_tm1 + z * z_out
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        xx = TT.dot(X, self.W) + self.b
        x_in = TT.dot(X, self.W_in) + self.b_in
        x_out = TT.dot(X, self.W_out) + self.b_out



        outputs, updates = theano.scan(
            self._step,
            sequences=[dict(input=xx,taps=[-1, -2]), dict(input=x_in,taps=[-1, -2]), dict(input=x_out, taps=[-1, -2])],
            outputs_info=[dict(initial=alloc_zeros_matrix(2, X.shape[1],  self.output_dim), taps=[-1, -2])],
            non_sequences=[self.U, self.U_in, self.U_out, self.bu_in, self.bu_out],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}


class SimpleGatedUnit2(Layer):

    def __init__(self, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='softplus', inner_activation='hard_sigmoid',
        gate_activation= 'tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(SimpleGatedUnit2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.gate_activation = activations.get(gate_activation)
        self.input = TT.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.W_gate = self.init((self.input_dim, self.output_dim))
        self.b_gate = shared_zeros((self.output_dim))
        self.U_gate = self.inner_init((self.output_dim, self.output_dim))

        self.params = [
            self.W, self.U, self.b,
            self.W_gate, self.b_gate,
            self.U_gate
        ]

        if weights is not None:
            self.set_weights(weights)



    def disp_var(self, name, value, index):
        return TT.cast(theano.printing.Print(name + str(index))(value[index]) * 1e-6, 'float32')


    def _step(self,
        xx,
        x_gate,
        h_tm1,
        u, u_gate):
        z = self.inner_activation(xx + TT.dot(h_tm1, u))
        z_gate = self.gate_activation(TT.dot(x_gate * h_tm1, u_gate))
        z_out = self.activation(h_tm1 * z_gate)
        h_t = z * z_out + (1-z) * h_tm1
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))
        xx = TT.dot(X, self.W) + self.b
        x_gate = TT.dot(X, self.W_gate) + self.b_gate

        outputs, updates = theano.scan(
            self._step,
            sequences=[xx, x_gate],
            outputs_info=[alloc_zeros_matrix(X.shape[1],  self.output_dim)],
            non_sequences=[self.U, self.U_gate],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

class StackableSGU(SimpleGatedUnit2):
    def __init__(self, *args, **kwargs):
        super(StackableSGU, self).__init__(*args, **kwargs)

    def get_output(self, train):
        X = self.get_input(train)
        if X.ndim == 3:
            X = X.dimshuffle((1,0,2))
        x_t = TT.dot(X, self.W) + self.b
        x_gate = TT.dot(X, self.W_gate) + self.b_gate

        outputs, updates = theano.scan(
            self._step,
            sequences=[x_t, x_gate],
            outputs_info=[alloc_zeros_matrix(X.shape[1],  self.output_dim)],
            non_sequences=[self.U, self.U_gate],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

class SimpleGatedUnit3(Layer):

    def __init__(self, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='softplus', inner_activation='hard_sigmoid',
        gate_activation= 'tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(SimpleGatedUnit3, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.gate_activation = activations.get(gate_activation)
        self.input = TT.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.W_gate = self.init((self.input_dim, self.output_dim))
        self.b_gate = shared_zeros((self.output_dim))
        self.U_gate = self.inner_init((self.output_dim, self.output_dim))

        self.params = [
            self.W, self.U, self.b,
            self.W_gate, self.b_gate,
            self.U_gate
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self,
        xx, xx_tm1,
        x_gate,
        h_tm1,
        u, u_gate):
        z = self.inner_activation(xx + TT.dot(h_tm1, u))
        z_gate = self.gate_activation(TT.dot(x_gate * h_tm1, u_gate))
        z_out = self.activation(h_tm1 * z_gate)

        h_t = z * z_out + (1-z) * h_tm1
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))
        xx = TT.dot(X, self.W) + self.b
        x_gate = TT.dot(X, self.W_gate) + self.b_gate

        outputs, updates = theano.scan(
            self._step,
            sequences=[dict(input=xx,taps=[-1, -2]),dict(input=x_gate, taps=[-1])],
            outputs_info=[alloc_zeros_matrix(X.shape[1],  self.output_dim)],
            non_sequences=[self.U, self.U_gate],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

class ClockworkRNN(Layer):
    def __init__(self, periods, input_dim, output_dim=128,
        init='normal', inner_init='normal',
        activation='tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):

        self.periods = periods
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.input = TT.tensor3()

        self.n = self.output_dim // len(self.periods)


        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.periods = E.tools.asarray(sorted(self.periods))

        self.clock_weights = {}
        for i, period in enumerate(self.periods):
            self.clock_weights[period] = self.inner_init((
                (len(self.periods)-i) * self.n, self.n
            ))

        self.U = self.inner_init((self.output_dim, self.output_dim))

        self.params = [
        self.W, self.U,
        self.b,

        ]
        self.params.extend(self.clock_weights.values())

        assert self.output_dim % len(self.periods) == 0

        super(ClockworkRNN, self).__init__()


    """
        def step(self, input_step, previous_activation, time_step, W_in, W_self, biases):
            new_activation = previous_activation.copy()
            modzero = TT.nonzero(TT.eq(TT.mod(time_step, self.group_labels), 0))[0]
            W_in_now = TT.flatten(W_in[:, modzero, :], outdim=2)
            W_self_now = TT.flatten(W_self[:, modzero, :], outdim=2)
            biases_now = TT.flatten(biases[modzero, :])
            activation = TT.dot(input_step, W_in_now)
            activation += TT.dot(previous_activation, W_self_now)
            activation += biases_now
            activation = self.activation_function(activation)
            modzero_activation_changes = (modzero * self.group_size) + (
                TT.ones((modzero.shape[0], self.group_size), dtype='int32') * TT.arange(self.group_size, dtype='int32')).T
            modzero_flatten = TT.flatten(modzero_activation_changes).astype('int32')
            new_activation = TT.set_subtensor(new_activation[:, modzero_flatten], activation)
            time_step += 1
            return new_activation, time_step
    """
    def _step(self, time, x_t, h_tm1):

        h_t = TT.concatenate([
            theano.ifelse.ifelse(
                TT.eq(time % period, 0),
                x_t[:, i*self.n:(i+1)*self.n] +
                 TT.dot(h_tm1[:, i*self.n::], self.clock_weights[period]),
                h_tm1[:, i*self.n:(i+1)*self.n])
                for i, period in enumerate(self.periods)], axis=1)
        return self.activation(h_t)

    def get_output(self, train):
        '''Transform inputs to this layer into outputs for the layer.
        Parameters
        ----------
        inputs : dict of theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`base.Layer.connect`.
        Returns
        -------
        outputs : dict of theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and a "hid" output that gives the post-activation values.
        updates : sequence of update pairs
            A sequence of updates to apply to this layer's state inside a theano
            function.
        '''
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))
        x = E.tools.TT.dot(X, self.W) + self.b

        outputs, updates = theano.scan(
            self._step,
            sequences=[E.tools.TT.arange(x.shape[0]), x],
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            truncate_gradient=self.truncate_gradient,
            )
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

class ClockworkGatedRNN(Layer):

    def __init__(self, periods, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='softplus', inner_activation='hard_sigmoid',
        gate_activation= 'tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(ClockworkGatedRNN, self).__init__()
        self.periods = periods
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.gate_activation = activations.get(gate_activation)

        self.n = self.output_dim // len(self.periods)

        assert self.output_dim % len(self.periods) == 0

        self.input = TT.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.W_gate = self.init((self.input_dim, self.output_dim))
        self.b_gate = shared_zeros((self.output_dim))


        self.clock_u = {}
        for i, period in enumerate(self.periods):
            self.clock_u[period] = self.inner_init((
                self.n, self.n
            ))

        self.clock_gates = {}
        for i, period in enumerate(self.periods):
            self.clock_gates[period] = self.inner_init((
                self.n, self.n

            ))



        self.params = [
            self.W, self.b,
        #    self.W_gate, self.b_gate,
        ]

        self.params.extend(self.clock_u.values())
        #self.params.extend(self.clock_gates.values())


        if weights is not None:
            self.set_weights(weights)

    def clock_gating(self, i, x_t, x_gate, h_tm1, u, u_gate):

        x_t_sub = x_t[:, i*self.n:(i+1)*self.n]
        x_gate_sub = x_t[:, i*self.n:(i+1)*self.n]
        h_tm1_sub = x_t[:, i*self.n:(i+1)*self.n]

        return self.gating(x_t_sub, x_gate_sub, h_tm1_sub, u, u_gate)

    def _step(self, time, x_t, x_gate, h_tm1):
        h_t = TT.concatenate([
            theano.ifelse.ifelse(
                TT.eq(time % period, 0),
                self.clock_gating(i, x_t, x_gate, h_tm1, self.clock_u[period], self.clock_gates[period]),
                h_tm1[:, i*self.n:(i+1)*self.n])
                for i, period in enumerate(self.periods)], axis=1)

        return h_t

    def _step_test(self, time, x_t, x_gate, h_tm1):
        h_t = TT.concatenate([
            theano.ifelse.ifelse(
                TT.eq(time % period, 0),
                self.clock_gating(i, x_t, x_gate, h_tm1, self.clock_u[period], self.clock_gates[period]),
                h_tm1[:, i*self.n:(i+1)*self.n])
                for i, period in enumerate(self.periods)], axis=1)

        return h_t

    def disp_var(self, name, value):
        return TT.cast(theano.printing.Print(name)(value) * 1e-6, 'float32')

    def disp_two_dims(self, name, value):
        return self.disp_var(name + " dim 0", value=value.shape[0]) + \
        self.disp_var(name + " dim 1", value=value.shape[1])

    def gating(self, x_t, x_gate, h_tm1, u, u_gate):
        k = TT.dot(h_tm1, u) + self.disp_two_dims("u", u)


        z = self.inner_activation(x_t + k) \
            + \
            self.disp_var(name="k",value=k) + \
            self.disp_var("k dim 0", value=k.shape[0]) + \
            self.disp_var("k dim 1", value=k.shape[1]) + \
            self.disp_var("x_t dim 0", value=x_t.shape[0]) + \
            self.disp_var("x_t dim 1", value=x_t.shape[1])


        z = z \
        + \
        self.disp_var("z dim 0", value=z.shape[0]) + \
        self.disp_var("z dim 1", value=z.shape[1])

        p = x_gate * h_tm1 \
        + \
        self.disp_var("x_gate dim 0", value=x_gate.shape[0]) + \
        self.disp_var("x_gate dim 1", value=x_gate.shape[1]) + \
        self.disp_var("h_tm1 dim 0", value=h_tm1.shape[0]) + \
        self.disp_var("h_tm1 dim 1", value=h_tm1.shape[1])


        u_gate = u_gate \
        +  \
        self.disp_var("u_gate dim 0", value=u_gate.shape[0]) + \
        self.disp_var("u_gate dim 1", value=u_gate.shape[1])

        q = TT.dot(p, u_gate) \
        + \
        self.disp_var("p dim 0", value=p.shape[0]) + \
        self.disp_var("p dim 1", value=p.shape[1])

        z_gate = self.gate_activation(q) \
        + \
        self.disp_var("q dim 0", value=q.shape[0]) + \
        self.disp_var("q dim 1", value=q.shape[1])

        z_gate = z_gate \
                 +  self.disp_two_dims("z_gate", z_gate)
        h_tm1 = h_tm1 \
                + self.disp_two_dims("h_tm1", h_tm1)

        s = h_tm1 * z_gate \
        + \
        self.disp_var("z_gate dim 0", value=z_gate.shape[0]) + \
        self.disp_var("z_gate dim 1", value=z_gate.shape[1])


        z_out = self.activation(s)
        h_t = z * z_out + (1-z) * h_tm1
        return h_t

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))
        x_t = TT.dot(X, self.W) + self.b
        x_gate = TT.dot(X, self.W_gate) + self.b_gate

        outputs, updates = theano.scan(
            self._step,
            sequences=[E.tools.TT.arange(x_t.shape[0]), x_t, x_gate],
            outputs_info=[alloc_zeros_matrix(X.shape[1],  self.output_dim)],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

class ClockworkGatedRNN1(Layer):
    def __init__(self, periods, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='softplus', inner_activation='hard_sigmoid',
        gate_activation= 'tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):

        self.periods = periods
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.gate_activation = activations.get(gate_activation)
        self.input = TT.tensor3()

        self.n = self.output_dim // len(self.periods)


        self.W = self.init((self.input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.W_gate = self.init((self.input_dim, self.output_dim))
        self.U_gate = self.inner_init((self.output_dim, self.output_dim))
        self.b_gate = shared_zeros((self.output_dim))

        self.periods = E.tools.asarray(sorted(self.periods))

        #self.clock_gates = {}
        #for i, period in enumerate(self.periods):
        #    self.clock_gates[period] = self.inner_init((
        #        (len(self.periods)-i) * self.n, self.n
        #    ))

        self.params = [
        self.W, self.U, self.b,
        self.W_gate, self.U_gate, self.b_gate,
        ]
        #self.params.extend(self.clock_gates.values())

        assert self.output_dim % len(self.periods) == 0

        super(ClockworkGatedRNN, self).__init__()


    def clock_gating(self, i , period, x_t, x_gate, h_tm1):


        x_t_sub = x_t[:, i*self.n:(i+1)*self.n]
        x_gate_sub = x_gate[:, i*self.n:(i+1)*self.n]
        h_tm1_sub = h_tm1[:, i*self.n::]
        u_gate = self.clock_gates[period]

        return self.gating(x_t_sub, x_gate_sub, h_tm1_sub, self.U, u_gate)

    def gating(self, x_t, x_gate, h_tm1, u, u_gate):
        z = self.inner_activation(x_t + TT.dot(h_tm1, u))
        z_gate = self.gate_activation(TT.dot(x_gate * h_tm1, u_gate))
        z_out = self.activation(h_tm1 * z_gate)
        h_t = z * z_out + (1-z) * h_tm1
        return h_t


    def _step(self, x_t, x_gate, h_tm1, u, u_gate):
        z = self.inner_activation(x_t + TT.dot(h_tm1, u))
        z_gate = self.gate_activation(TT.dot(x_gate * h_tm1, u_gate))
        z_out = self.activation(h_tm1 * z_gate)
        h_t = z * z_out + (1-z) * h_tm1
        return h_t

        #h_t = TT.concatenate([
        #    theano.ifelse.ifelse(
        #        TT.eq(time % period, 0),

                #self.gating(i, period, x_t, x_gate, h_tm1),
        #        self.gating(x_t, x_gate, h_tm1, self.U, self.clock_gates[period]),

        #        h_tm1[:, i*self.n:(i+1)*self.n])

         #       for i, period in enumerate(self.periods)], axis=1)

        #return self.gating(x_t, x_gate, h_tm1, u, u_gate)

    def get_output(self, train):

        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))
        x = E.tools.TT.dot(X, self.W) + self.b
        x_gate = E.tools.TT.dot(X, self.W_gate) + self.b_gate

        outputs, updates = theano.scan(
            self._step,
            sequences=[x, x_gate],
            outputs_info=alloc_zeros_matrix(X.shape[1], self.output_dim),
            non_sequences=[self.U, self.U_gate],
            truncate_gradient=self.truncate_gradient,


            )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "inner_activation":self.inner_activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}