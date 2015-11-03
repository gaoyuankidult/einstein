# training with current energy, force and energy derivative using neuron network

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import code

from numpy import array, vstack

'''
    Train a simple deep NN on the MNIST dataset.
    Get to 98.30% test accuracy after 20 epochs (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_epoch = 1000

import pickle
forces = array(pickle.load(open("forces_train.pickle","rb")))
diffslow = array(pickle.load(open("diffslow_train.pickle","rb")))
sfa = array(pickle.load(open("sfa_train.pickle","rb")))

print(forces.shape)
print(sfa.shape)
print(array([forces, sfa]).shape)
X_train = vstack([forces, sfa]).T
Y_train = diffslow * 10000


X_train = X_train.astype("float32")
Y_train = Y_train.astype("float32")

model = Sequential()
model.add(Dense(128, input_shape=(2,)))
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer="adam")

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
score = model.evaluate(X_train, Y_train, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


code.interact(local=locals())



