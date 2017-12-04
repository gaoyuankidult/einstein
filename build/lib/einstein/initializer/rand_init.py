__author__ = 'gao'

__author__ = 'yuangao'

from theano import config
from numpy import zeros
from numpy import minimum


class RandomSparseInit():

    def __init__(self):
        pass

    @classmethod
    def weight_init(
            cls,
            rng,
            size_x,
            size_y,
            sparsity,
            scale=0.01,
            ):

        """
        This function randomly generate the weights of matrix within a range.

        :type size_x: int
        :param size_x: first dimension of matrix (row number)

        :type size_y: int
        :param size_y: second dimension of matrix (column number)

        :type scale: float
        :param scale: scale of  standard deviation of Gaussian from which wights are sampled.

        :type sparsity: number of nodes you would like to make them non-zero
        :param sparsity: int


        :param rng: numpy random generator
        :type rng: numpy.random.RandomStates

        :return: None
        :rtype: None
        """
        assert type(size_x) is int, "size_x is not of type int"
        assert type(size_y) is int, "size_y is not of type int"
        if sparsity < 0:
            sparsity = size_y
        else:
            sparsity = minimum(size_y,sparsity)
        values = zeros([size_x, size_y],dtype=config.floatX)
        for dx in xrange(size_x):
            perm = rng.permutation(size_y)
            new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
            values[dx, perm[:sparsity]] = new_vals
        return values.astype(config.floatX)

    @classmethod
    def bias_init(cls,
                        rng,
                        size_x,
                        sparsity,
                        scale=0.01,
                        ):
        assert type(size_x) is int, "size_x is not of type int"
        if sparsity < 0:
            sparsity = size_x
        else:
            sparsity = minimum(size_x,sparsity)
        values = zeros([size_x],dtype=config.floatX)
        perm = rng.permutation(size_x)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[perm[:sparsity]] = new_vals
        return values.astype(config.floatX)
