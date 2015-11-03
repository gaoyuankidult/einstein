from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from Queue import PriorityQueue


from einstein.databank import imdb
from einstein.preprocessing import sequence
from einstein.models import Sequential
from einstein.layers.embedding import Embedding
from einstein.regularizers import Dropout
from einstein.layers import Dense
from einstein.layers.activations import Activation
from einstein.layers import SimpleDeepGRU, GRU, DeepGRU
from einstein.tools import np_utils

import einstein as E

depth_set = range(2)
init_set = ["uniform", "normal", "lecun_uniform", "glorot_normal", "glorot_uniform",  "he_normal", "he_uniform", "orthogonal", "zero"]
inner_init_set = ["uniform", "normal", "lecun_uniform", "glorot_normal", "glorot_uniform",  "he_normal", "he_uniform", "orthogonal", "zero", "identity"]
activation_set =  ["softmax", "time_distributed_softmax", "softplus", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
inner_activation_set = ["softmax", "time_distributed_softmax", "softplus", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]

def test(depth, init, inner_init, activation, inner_activation):
    max_features=20000
    maxlen = 100 # cut texts after this number of words (among top max_features most common words)
    batch_size = 16
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')


    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 256))
    model.add(GRU(256, 128, inner_init=inner_init, inner_activation=inner_activation, init=init, activation=activation))
    model.add(Dense(128, 1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    print("Train...")
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5, validation_split=0.1, show_accuracy=True, verbose=0)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)

    classes = model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, y_test)
    print('Test accuracy:', acc)
    return acc, history

if __name__ == "__main__":
    import itertools
    a = [depth_set, init_set, inner_init_set, activation_set, inner_activation_set]
    all_combs = list(itertools.product(*a))

    best_acc = 0
    best_comb = []
    fileio = E.fileio.FileIO("record_parameter_gru.txt")

    for depth, init, inner_init, activation, inner_activation in all_combs:
        print((depth, init, inner_init, activation, inner_activation))
        fileio.save_line("current comb:")
        fileio.save_line((depth, init, inner_init, activation, inner_activation))
        try:
            acc, history = test(depth, init, inner_init, activation, inner_activation)
            print(history)
            if acc > max(history['val_acc']) :
                    pass
            else:
                acc = max(history['val_acc'])
        except Exception as e:
            print(e)
            acc = 0


        fileio.save_line("current best acc:")
        fileio.save_line(acc)
        if acc > best_acc:
            best_comb = [depth, init, inner_init, activation, inner_activation]
            best_acc = acc
            print("current best comb:", (best_comb), "acc:", best_acc)
        else:
             pass