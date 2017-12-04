from __future__ import absolute_import
import six.moves.cPickle
import gzip
import random
from six.moves import zip
import numpy as np
import zipfile


def load_data(path="trainingandtestdata.zip", nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):

    path = get_file(path, origin="http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    elif path.endswith(".zip"):
        f = zipfile.open(path, 'rb')
    else:
        f = open(path, 'rb')

