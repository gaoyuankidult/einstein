from numpy import array, arange

import matplotlib.pyplot as p

from pickle import load

from .. tools import AbstractMethod


class Plot(object):
    def __init__(self, name, figsize=None):
        self.p = p
        self.fig = self.p.figure(figsize=figsize)
        self.name = name
        self._x_label = None
        self._y_label = None
        self._title = None

    def plot(self):
        AbstractMethod()

    @property
    def xlabel(self):
        return self._x_label

    @xlabel.setter
    def xlabel(self, value):
        print value
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._x_label = value
        self.p.xlabel(value)

    @property
    def ylabel(self):
        return self.xlabel

    @ylabel.setter
    def ylabel(self, value):
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._y_label = value
        self.p.ylabel(value)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._title = value
        self.fig.suptitle(value)


class BoxPlot(Plot):
    def __init__(self, name = "boxplot", figsize=None):
        assert  isinstance(name, basestring)
        super(BoxPlot, self).__init__(name, figsize)

    def plot(self, data, x_labels=None):
        self.ax = self.fig.add_subplot(111)
        self.fig.suptitle(self._title)
        self.add_plot(data, x_labels)

    def add_plot(self, data, x_labels=None):
        self.ax.boxplot(data)
        if x_labels is not None:
            self.ax.set_xticks(arange(len(x_labels))+1)
            self.ax.set_xticklabels(x_labels, rotation=-45)

    def save_fig(self):
        self.p.savefig(self.name + ".png")

    def show(self):
        self.p.show()

    def plot_from_array(self, data, x_labels=None):
        self.plot(data=data, x_labels=x_labels)

    def plot_from_file(self, filename, x_labels=None):
        data = array(load(open(filename, "rb")))
        self.plot(data=data, x_labels=x_labels)

        


