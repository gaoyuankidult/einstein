from __future__ import print_function
from keras.layers.recurrent import LSTM, GRU, Layer ,SimpleRNN
from keras import activations, initializers
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

class SGU(Layer):

    def __init__(self, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='softplus', inner_activation='hard_sigmoid',
        gate_activation= 'tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(SGU, self).__init__()
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

class StackableSGU(SGU):
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

class SGUModified1(SGU):
    def __init__(self, *args, **kwargs):
        super(SGUModified1, self).__init__(*args, **kwargs)
        #self.U_gate2 = self.inner_init((self.output_dim, self.output_dim))

        #self.params.extend([self.U_gate2])

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))
        x_t = TT.dot(X, self.W) + self.b
        x_gate = TT.dot(X, self.W_gate) + self.b_gate

        outputs, updates = theano.scan(
            self._step,
            sequences=[x_t, x_gate],
            outputs_info=[dict(initial=alloc_zeros_matrix(3,  X.shape[1],  self.output_dim), taps=[-1, -2, -3])],
            non_sequences=[self.U, self.U_gate],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def _step(self,
        x_t,
        x_gate,
        h_tm1, h_tm2, h_tm3,
        u, u_gate):
        h = (h_tm1 + h_tm2 + h_tm3)/3.

        z = self.inner_activation(x_t + TT.dot(h, u))

        z_gate = self.gate_activation(TT.dot(h * x_gate, u_gate))
        z_out = self.activation(h * z_gate)

        h_t = z * z_out + (1-z) * h
        return h_t

class DSGU(SGU):
    def __init__(self, *args, **kwargs):
        super(DSGU, self).__init__(*args, **kwargs)
        self.sig= activations.get("sigmoid")
        self.tanh = activations.get("tanh")
        self.U_gate2 = self.inner_init((self.output_dim, self.output_dim))
        self.params.extend([self.U_gate2])

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))


        xx = TT.dot(X, self.W) + self.b
        x_gate = TT.dot(X, self.W_gate) + self.b_gate


        outputs, updates = theano.scan(
            self._step,
            sequences=[xx, x_gate],
            outputs_info=[alloc_zeros_matrix(X.shape[1],  self.output_dim)],
            non_sequences=[self.U, self.U_gate, self.U_gate2],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def _step(self,
        x_t,
        x_gate,
        h_tm1,
        u, u_gate, u_gate2):
        z = self.inner_activation(x_t + TT.dot(h_tm1, u))

        z_gate = self.tanh(TT.dot(x_gate * h_tm1, u_gate))
        z_out = self.sig(TT.dot(z_gate * h_tm1, u_gate2))

        h_t = z * z_out + (1-z) * h_tm1
        return h_t

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
                (i+1) * self.n, self.n
            ))

        self.params = [
        self.W,
        self.b,

        ]
        self.params.extend(self.clock_weights.values())

        assert self.output_dim % len(self.periods) == 0

        super(ClockworkRNN, self).__init__()

    def _step(self, time, x_t, h_tm1):

        h_t = TT.concatenate([
            theano.ifelse.ifelse(
                TT.eq(time % period, 0),
                x_t[:, i*self.n:(i+1)*self.n] +
                 TT.dot(h_tm1[:, :(i+1)*self.n], self.clock_weights[period]),
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

class ClockworkGRU(Layer):

    def __init__(self, periods, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='sigmoid', inner_activation='sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(ClockworkGRU, self).__init__()
        self.periods = periods
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.n = self.output_dim // len(self.periods)

        #assert self.output_dim % len(self.periods) == 0

        self.input = TT.tensor3()

        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.Wr = self.init((self.input_dim, self.output_dim))
        self.br = shared_zeros((self.output_dim))

        self.Wz = self.init((self.input_dim, self.output_dim))
        self.bz = shared_zeros((self.output_dim))

        self.clock_h = {}
        for i, period in enumerate(self.periods):
            self.clock_h[period] = self.inner_init((
                (i + 1) * self.n, self.n
            ))

        self.clock_rgates = {}
        for i, period in enumerate(self.periods):
            self.clock_rgates[period] = self.inner_init((
                (i + 1) * self.n, (i + 1) * self.n

            ))

        self.clock_zgates = {}
        for i, period in enumerate(self.periods):
            self.clock_zgates[period] = self.inner_init((
                (i + 1) * self.n, self.n

            ))


        self.params = [
            self.W, self.b,
            self.Wr, self.br,
            self.Wz, self.bz
        ]

        self.params.extend(self.clock_h.values())
        self.params.extend(self.clock_rgates.values())
        self.params.extend(self.clock_zgates.values())


        if weights is not None:
            self.set_weights(weights)


    def inner_fn(self, T, x_t, r_t, z_t, h_tm1, nah_tm1):
        r = TT.nnet.sigmoid(r_t + TT.dot(h_tm1, self.clock_rgates[T]))
        z = TT.nnet.sigmoid(z_t + TT.dot(h_tm1, self.clock_zgates[T]))
        pre = x_t + TT.dot(r * h_tm1, self.clock_h[T])

        h_t = self.activation(pre)

        v1 = z * h_t
        v2 = (1 - z) * nah_tm1
        v = v1 + v2
        return v

    def _step(self, time, x_t, x_rt, x_zt, h_tm1):
        h_t = TT.concatenate([
            theano.ifelse.ifelse(
                TT.eq(time % period, 0),
                self.inner_fn(period, x_t[:, i* self.n:(i+1)* self.n], x_rt[:, :(i+1)* self.n], x_zt[:, i*self.n:(i+1)*self.n], h_tm1[:, :(i+1)*self.n], h_tm1[:, i*self.n:(i+1)*self.n]),
                h_tm1[:, i*self.n:(i+1)*self.n])
                for i, period in enumerate(self.periods)], axis=1)
        return h_t


    def disp_var(self, name, value):
        return TT.cast(theano.printing.Print(name)(value) * 1e-6, 'float32')

    def disp_two_dims(self, name, value):
        return self.disp_var(name + " dim 0", value=value.shape[0]) + \
        self.disp_var(name + " dim 1", value=value.shape[1])

    def get_output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))
        x_t = TT.dot(X, self.W) + self.b
        x_rt = TT.dot(X, self.Wr) + self.br
        x_zt = TT.dot(X, self.Wz) + self.bz




        outputs, updates = theano.scan(
            self._step,
            sequences=[E.tools.TT.arange(x_t.shape[0]), x_t, x_rt, x_zt],
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

class ClockworkSGU(Layer):

    def __init__(self, periods, input_dim, output_dim=128,
        init= 'uniform', inner_init='glorot_normal',
        activation='softplus', inner_activation='hard_sigmoid',
        gate_activation= 'tanh',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(ClockworkSGU, self).__init__()
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


        self.clock_h = {}
        for i, period in enumerate(self.periods):
            self.clock_h[period] = self.inner_init((
                (i + 1) * self.n, self.n
            ))


        self.clock_gates = {}
        for i, period in enumerate(self.periods):
            self.clock_gates[period] = self.inner_init((
                (i + 1) * self.n, self.n

            ))


        self.params = [
            self.W, self.b,
            self.W_gate, self.b_gate,
        ]

        self.params.extend(self.clock_h.values())
        self.params.extend(self.clock_gates.values())


        if weights is not None:
            self.set_weights(weights)


    def inner_fn(self, T, x_t, x_gate, h_tm1, nah_tm1):
        z = self.inner_activation(x_t + TT.dot(h_tm1, self.clock_h[T]))
        z_gate = self.gate_activation(TT.dot(x_gate * h_tm1, self.clock_gates[T]))

        z_out = self.activation(nah_tm1 * z_gate)
        h_t = z * z_out + (1-z) * nah_tm1


        return h_t

    def _step(self, time, x_t, x_gate, h_tm1):

        h_t = TT.concatenate([
            theano.ifelse.ifelse(
                TT.eq(time % period, 0),
                self.inner_fn(period, x_t[:, i* self.n:(i+1)* self.n], x_gate[:, :(i+1)* self.n], h_tm1[:, :(i+1)*self.n], h_tm1[:, i*self.n:(i+1)*self.n]),
                h_tm1[:, i*self.n:(i+1)*self.n])
                for i, period in enumerate(self.periods)], axis=1)
        return h_t

    def disp_var(self, name, value):
        return TT.cast(theano.printing.Print(name)(value) * 1e-6, 'float32')

    def disp_two_dims(self, name, value):
        return self.disp_var(name + " dim 0", value=value.shape[0]) + \
        self.disp_var(name + " dim 1", value=value.shape[1])

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

