from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from einstein.layers.recurrent import LSTM, GRU, SGU
from keras.datasets.data_utils import get_file
import einstein as E
import numpy as np
import random, sys
import keras
import time
from einstein.fileio import FileIO

'''
    Example script to generate text from Nietzsche's writings.
    At least 20 epochs are required before the generated text
    starts sounding coherent.
    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.
    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)))
y = np.zeros((len(sentences), len(chars)))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1.
    y[i, char_indices[next_chars[i]]] = 1.


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(SGU(input_dim=len(chars), output_dim=512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(512, len(chars)))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a)-1)
        if a[i] > random.random():
            return i

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.times = []
        self.start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time()-self.start_time)
        self.val_losses.append(logs.get("val_acc"))
history = LossHistory()

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1, show_accuracy=True, validation_split = 0.1, callbacks=[history])

record_file = FileIO()
record_file.save_pickle(history.val_losses, "single_sgu_adam_sigmoid_titan_x_losses" )
record_file.save_pickle(history.times, "single_sgu_adam_sigmoid_titan_x_times" )
