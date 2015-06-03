import numpy as np
import theano
from keras.models import Sequential



class DeepQNetwork(Sequential):
    def __init__(self):
        self._compute_q_vals = \
            theano.function([self.q_layers[0].input_var],
            self.q_layers[-1].predictions(),
            on_unused_input='ignore')


    def choose_action(self, state, epsilon):
        """
        Choose a random action with probability epsilon,
        or return the optimal action.
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.q_vals(state))


    def q_vals(self, state):
        """ Return an array of q-values for the indicated state (phi)
        """
        state_batch = np.zeros((self._batch_size,
                          self._phi_length,
                          self._img_height,
                          self._img_width), dtype=theano.config.floatX)
        state_batch[0, ...] = state
        return self._compute_q_vals(state_batch)[0, :]