from __future__ import absolute_import
from __future__ import print_function


import theano
import theano.tensor as T
import numpy as np
import warnings

from keras import optimizers
from keras import objectives
from keras import regularizers
from keras import constraints
from keras.models import Sequential
from keras import callbacks as cbks
import time, copy
from keras.utils.generic_utils import Progbar, printv
from six.moves import range


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def standardize_weights(y, sample_weight=None, class_weight=None):
    if sample_weight is not None:
        return standardize_y(sample_weight)
    elif isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise Exception('class_weight not supported for 3+ dimensional targets.')
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        return np.expand_dims(np.array(list(map(lambda x: class_weight[x], y_classes))), 1)
    else:
        return np.ones(y.shape[:-1] + (1,))

def standardize_y(y):
    if not hasattr(y, 'shape'):
        y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)
    return y

def standardize_X(X):
    if type(X) == list:
        return X
    else:
        return [X]


def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]

class Sequential(Sequential):

    def __init__(self,*args, **kwargs):
        super(Sequential, self).__init__(*args, **kwargs)

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
        validation_split=0., validation_data=None, shuffle=True, show_accuracy=False,
        class_weight=None, sample_weight=None):

        X = standardize_X(X)
        y = standardize_y(y)
        sample_weight = standardize_weights(y, class_weight=class_weight, sample_weight=sample_weight)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test
        if validation_data:
            try:
                X_val, y_val = validation_data
            except:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val). \
                    X_val may be a numpy array or a list of numpy arrays depending on your model input.")
            X_val = standardize_X(X_val)
            y_val = standardize_y(y_val)

            self.X_val = X_val
            self.y_val = y_val

            val_ins = X_val + [y_val, np.ones(y_val.shape[:-1] + (1,))]

        if show_accuracy:
            f = self._train_with_acc
            out_labels = ['loss', 'acc']
        else:
            f = self._train
            out_labels = ['loss']

        ins = X + [y, sample_weight]
        metrics = ['loss', 'acc', 'val_loss', 'val_acc']
        return self._fit(f, ins, out_labels=out_labels, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=callbacks, \
            validation_split=validation_split, val_f=val_f, val_ins=val_ins, shuffle=shuffle, metrics=metrics)

    def _fit(self, f, ins, out_labels=[], batch_size=128, nb_epoch=100, verbose=1, callbacks=[], \
        validation_split=0., val_f=None, val_ins=None, shuffle=True, metrics=[]):

        '''
            Abstract fit function for f(*ins). Assume that f returns a list, labelled by out_labels.
        '''
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print("Train on %d samples, validate on %d samples" % (len(ins[0]), len(val_ins[0])))
        else:
            if 0 < validation_split < 1:
                do_validation = True
                split_at = int(len(ins[0]) * (1 - validation_split))
                (ins, val_ins) = (slice_X(ins, 0, split_at), slice_X(ins, split_at))
                if verbose:
                    print("Train on %d samples, validate on %d samples" % (len(ins[0]), len(val_ins[0])))

        nb_train_sample = len(ins[0])
        index_array = np.arange(nb_train_sample)

        history = cbks.History()
        if verbose:
            callbacks = [history, cbks.BaseLogger()] + callbacks
        else:
            callbacks = [history] + callbacks
        callbacks = cbks.CallbackList(callbacks)

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics':metrics,
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                ins_batch = slice_X(ins, batch_ids)

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(*ins_batch)
                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                # do validation after each batch as well
                if do_validation:
                    val_loss, val_acc = self.test(self.X_val, self.y_val, accuracy=True)
                    batch_logs['val_loss'] =  val_loss
                    batch_logs['val_acc'] =  val_acc

                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1: # last batch
                    # validation
                    epoch_logs = {}
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins, batch_size=batch_size, verbose=0)
                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return history