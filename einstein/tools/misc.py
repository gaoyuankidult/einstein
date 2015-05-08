import theano as T
import unittest
from collections import Sequence
from itertools import chain, count
from numpy import array
from numpy import zeros
from pickle import dump
from pickle import load


def theano_form(data, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param data: list to be transformed
    :param shape: output shape
    :return:
    """
    return array(data, dtype=T.config.floatX).reshape(shape)

def array_form(data):
    """
    This function transfer any list to a form that has T.config.floatX as datatype
    :param list:
    :return:
    """
    return array(data, dtype=T.config.floatX)

def check_list_depth(seq):
    seq = iter(seq)
    try:
        for level in count():
            seq = chain([next(seq)], seq)
            seq = chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level

def make_chunks(lst, n):
    """ Yield successive n-sized chunks from l.
    """
    dim = len(lst[0])
    l = ([[0] * dim] * (n-1))
    l.extend(lst)
    for i in xrange(0, len(l)- n + 1, 1):
        yield l[i:i+n]


def save_to_pickle_file(data, file_name):
    with open('%s.pickle' % file_name, 'wb') as handle:
        dump(data, handle)

def load_pickle_file(file_name):
    with open('%s.pickle' % file_name, 'rb') as handle:
        data = load(handle)
    return data

def running_cumsum(data, n):
    """
    :param data: an array
    :param n: number of running average
    :return: array that stored running average
    """
    total_average_data_length = int(len(data)/float(n))
    final_data = zeros(total_average_data_length)
    for i in xrange(n):
        final_data += data[i::n]
    return final_data/float(n)

def check_none_join(value):
    """
    check whether value is none or not. If it is, then do nothing else delete
    :param value:
    :return:
    """
    if value is None:
        pass
    else:
        value.join()


class TestPickleMethods(unittest.TestCase):

    def test_pickle(self):
        """
        Test two functions, both save and load using pickle format
        :return:
        """
        data = [1, 2, 3]
        file_name = "pickle_test"
        save_to_pickle_file(data, file_name)
        self.assertEqual(data, load_pickle_file(file_name))



if __name__ == "__main__":
    unittest.main()