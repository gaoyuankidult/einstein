from numpy import array, arange

import matplotlib.pyplot as p

from pickle import load

from .. tools import AbstractMethod


class Plot(object):
    def __init__(self, name, plt, figsize=None):
        if plt == None:
            self.p = p
        else:
            self.p = plt
        self.fig = self.p.figure(figsize=figsize)
        self.name = name
        self._x_label = None
        self._y_label = None
        self._title = None

    def plot(self):
        AbstractMethod()

    def xlabel(self, value):
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._x_label = value
        self.p.xlabel(value)

    def ylabel(self, value):
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._y_label = value
        self.p.ylabel(value)

    def title(self, value):
        assert isinstance(value, basestring), "value: %s is not of basestring type." % value
        self._title = value
        self.fig.suptitle(value)

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