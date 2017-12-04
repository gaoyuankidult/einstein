__author__ = 'gao'

from theano import config
from theano import shared
from theano import grad
from numpy import zeros
from numpy import float32
from numpy import cast
from collections import OrderedDict
import theano as T
import theano.tensor as TT

def nesterov_grad(params,
                  grads,
                  updates,
                  learning_rate = 1e-3,
                  momentum=0.6,
                  weight_decay=0.01):

    for param_i, grad_i in zip(params, grads):
        mparam_i = shared(zeros(param_i.get_value().shape, dtype=config.floatX))
        full_grad = grad_i + weight_decay * param_i
        v = momentum * mparam_i - learning_rate * full_grad # new momemtum
        w = param_i + momentum * v - learning_rate * full_grad # new parameter values
        updates.append((mparam_i, v))
        updates.append((param_i, w))


def adam_grad(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
         epsilon=1e-8, gamma=1-1e-8):
    updates = []
    all_grads = T.grad(loss,all_params)

    i = T.shared(float32(1))  # HOW to init scalar shared?
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED
    learning_rate_t = learning_rate * (TT.sqrt(fix2) / fix1)

    for param_i, g in zip(all_params, all_grads):
        m = T.shared(
            zeros(param_i.get_value().shape, dtype=T.config.floatX))
        v = T.shared(
            zeros(param_i.get_value().shape, dtype=T.config.floatX))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m) # CHANGED from b_t to use beta1_t
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (TT.sqrt(v_t) + epsilon)
        param_i_t = param_i - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t) )
    updates.append((i, i_t))
    return updates
