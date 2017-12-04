from einstein.models import Sequential
from einstein.layers import Convolution2D, Flatten, Dense
from einstein.layers.activations import Activation

class VisuoMotor(Sequential):
    def __init__(self):
        super(VisuoMotor, self).__init__()


if __name__ == "__main__.py":
    model = VisuoMotor()

    model.add(Convolution2D(nb_filter=64, stack_size=3, nb_row=7, nb_col=7))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=32, stack_size=64, nb_row=5, nb_col=5))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=32, stack_size=32, nb_row=5, nb_col=5))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=32, stack_size=32, nb_row=5, nb_col=5))

    model.add(Flatten())
    model.add(Dense(32*5*5, 64, init='normal'))
    model.add(Activation('relu'))

    model.add(Dense(64, 40, init='normal'))
    model.add(Activation('relu'))

    model.add(Dense(40, 40, init='normal'))
    model.add(Activation('relu'))

    model.add(Dense(40, 7, init='normal'))
