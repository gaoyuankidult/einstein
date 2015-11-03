from numpy import array, arange

import matplotlib.pyplot as p

from pickle import load

from .. tools import AbstractMethod


class Plot(object):
<<<<<<< HEAD
    def __init__(self, name, plt, figsize=None):
        if plt == None:
            self.p = p
        else:
            self.p = plt
=======
    def __init__(self, name, figsize=None):
        self.p = p
>>>>>>> 21c5a6a5b15be02716243cb1dc1065a0e8ee4ce2
        self.fig = self.p.figure(figsize=figsize)
        self.name = name
        self._x_label = None
        self._y_label = None
        self._title = None

    def plot(self):
        AbstractMethod()

<<<<<<< HEAD
    def xlabel(self, value):
=======
    @property
    def x_label(self):
        return self._x_label

    @x_label.setter
    def x_label(self, value):
        print value
>>>>>>> 21c5a6a5b15be02716243cb1dc1065a0e8ee4ce2
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._x_label = value
        self.p.xlabel(value)

<<<<<<< HEAD
    def ylabel(self, value):
=======
    @property
    def y_label(self):
        return self.x_label

    @y_label.setter
    def y_label(self, value):
>>>>>>> 21c5a6a5b15be02716243cb1dc1065a0e8ee4ce2
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._y_label = value
        self.p.ylabel(value)

<<<<<<< HEAD
=======
    @property
    def title(self):
        return self._title

    @title.setter
>>>>>>> 21c5a6a5b15be02716243cb1dc1065a0e8ee4ce2
    def title(self, value):
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._title = value
        self.fig.suptitle(value)

<<<<<<< HEAD
    def plot_ins(self):
        return self.p

    def close(self):
        self.p.close()


class BoxPlot(Plot):
    def __init__(self, name = "boxplot",  plt = None, figsize=None):
        assert  isinstance(name, basestring)
        super(BoxPlot, self).__init__(name, plt, figsize)

    def plot(self, data, x_labels=None, label_step=1, label_shift=1):
        self.label_shift = label_shift
        self.label_step= label_step
        self.ax = self.fig.add_subplot(111)
        self.fig.suptitle(self._title)
        self.add_plot(data, x_labels)

    def add_plot(self, data, x_labels=None):
        self.ax.boxplot(data)
        if x_labels is not None:
            x_labels = map(lambda x: "%.2f" % x, x_labels)
            self.ax.set_xticks(arange(len(x_labels), step=self.label_step) + self.label_shift)
            self.ax.set_xticklabels(x_labels, rotation=-45)

    def savefig(self, name):
        self.p.savefig(name)

    def show(self):
        self.p.show()

    def plot_from_array(self, data, x_labels=None):
        self.plot(data=data, x_labels=x_labels)

    def plot_from_file(self, filename, x_labels=None):
        data = array(load(open(filename, "rb")))
        self.plot(data=data, x_labels=x_labels)

    def clf(self):
        self.p.clf()

    def figure(self, number):
        self.p.figure(number)
=======

class BoxPlot(Plot):
    def __init__(self, name = "boxplot", figsize=None):
        assert  isinstance(name, basestring)
        super(BoxPlot, self).__init__(name, figsize)

    def _plot(self, data, x_labels=None):
        ax = self.fig.add_subplot(111)
        self.fig.suptitle(self._title)
        ax.boxplot(data)
        if x_labels is not None:
            ax.set_xticks(arange(len(x_labels))+1)
            ax.set_xticklabels(x_labels, rotation=-45)
        self.p.savefig(self.name + ".png")
        self.p.show()

    def plot_from_array(self, data, x_labels=None):
        self._plot(data=data, x_labels=x_labels)

    def plot(self, filename, x_labels=None):
        data = array(load(open(filename, "rb")))
        print data
        self._plot(data=data, x_labels=x_labels)

        



>>>>>>> 21c5a6a5b15be02716243cb1dc1065a0e8ee4ce2
