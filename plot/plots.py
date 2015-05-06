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
    def x_label(self):
        return self._x_label

    @x_label.setter
    def x_label(self, value):
        print value
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._x_label = value
        self.p.xlabel(value)

    @property
    def y_label(self):
        return self.x_label

    @y_label.setter
    def y_label(self, value):
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

        



