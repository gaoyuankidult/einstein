import lasagne as L
import theano as T
import theano.tensor as TT
from ..tools import AbstractMethod
from ..layers import LSTMLayer, ReshapeLayer, InputLayer, DenseLayer
import einstein as E
from collections import namedtuple

class Model(object):
    """
    This class takes care of the modelling of system.
    """
    def __init__(self, setting, name="not given", default="default"):
        """
        This function is responsible for creating a model based on the dictionary
        :param model_params: a ordered dictionary type that contains information of layers.
        First item in dictionary is the first layer. Second item in dictionary is the second layer.
        :type model_params: collections.OrderedDict
        :return: None
        :rtype: None
        """
        # Get input as internal representation
        self.setting = setting
        self.name = name
        self.model_params = setting.get_layers_params()

        # Initialize symbolic parameters
        self.__init_symb()

        # Initialize parameters needed for later calculation
        self.current_layer = None
        self.previous_layer = None

        # Initialize communication socket
        self.ring_buffer = E.tools.RingBuffer(size=self.setting.n_time_steps + 1)  # need reward of next step for training

        # Build the model
        if default == "default":
            self.build_default_model()
            self.build_functions()
        elif default == "masked":
            self.build_masked_model()
            self.build_masked_functions()



    def __init_symb(self):
        """
        Initialize the symbolic variables of the model (e.g. input and output)
        :return:
        """
        self.input = TT.tensor3('input')
        self.target_output = TT.tensor3('target_output')
        self.mask = TT.matrix("mask")

    def save_all_params(self, file_name):
        list_of_all_params = []
        for layer in self.layers:
            try:
                list_of_all_params.append(layer.get_weight_params())
            except AttributeError as e:
                list_of_all_params.append([])
        E.tools.save_to_pickle_file(list_of_all_params, file_name)

    def load_all_params(self, file_name):
        AbstractMethod()

    def build_model(self):
        AbstractMethod()

    def build_masked_model(self):
        self.__build_network()
        self.__build_masked_cost_function()
        self.__build_training_rule()
        print "Model: %s built according to masked setting, with input data and output data as two input variables..." \
              % (self.name)


    def build_default_model(self):
        self.__build_network()
        self.__build_cost_function()
        self.__build_training_rule()
        print "Model: %s built according to default setting, with input data and output data as two input variables..." \
              % (self.name)

    def __build_network(self):
        """
        Use parameters of the layers to build model.
        :return:
        """
        # Parameters for storing all layers
        self.layers = []

        # Instantiate all layer
        for layer, params in self.model_params:
            # Test whether this is the first layer or not

            if type(layer) == InputLayer:
                self.current_layer = layer(**params)
            elif None == self.previous_layer:
                self.current_layer = layer(**params)
            elif layer == LSTMLayer \
                or layer == ReshapeLayer:
                self.current_layer = layer(input_layer=self.previous_layer, **params)
            else:
                self.current_layer = layer(incoming=self.previous_layer, **params)

            # Append current layer into layer list
            self.layers.append(self.current_layer)
            self.previous_layer = self.current_layer

    def __build_masked_cost_function(self):
        if self.setting._cost_f == None:
            self.cost = TT.mean((self.layers[-1].get_output(self.input, mask=self.mask)[:, :, :]
                    - self.target_output[:, :, :])**2)
        else:
            self.cost = self.cost_f(self.layers[-1].get_output(self.input, mask=self.mask)[:, :, :], self.target_output[:, :, :])


    def __build_cost_function(self):
        if self.setting._cost_f == None:
            self.cost = TT.mean((self.layers[-1].get_output(self.input)[:, :, :]
                    - self.target_output[:, :, :])**2)
        else:
            self.cost = self.cost_f(self.layers[-1].get_output(self.input)[:, :, :], self.target_output[:, :, :])

    def __build_training_rule(self):
        # Use NAG for training
        all_params = L.layers.get_all_params(self.layers[-1])
        self.updates = E.optimizer.adam_grad(self.cost, all_params, self.setting.learning_rate)

    def build_masked_functions(self):
        self._train = T.function([self.input, self.target_output, self.mask], self.cost, updates=self.updates,on_unused_input='ignore')
        self.predict = T.function([self.input, self.mask], self.layers[-1].get_output(self.input, mask=self.mask), on_unused_input='ignore')
        self.compute_cost = T.function([self.input, self.target_output, self.mask], self.cost, on_unused_input='ignore')
        #self.compute_cost = T.function([self.input, self.target_output, self.mask], self.cost, on_unused_input='ignore')

    def build_functions(self):
        self._train = T.function([self.input, self.target_output], self.cost, updates=self.updates)
        self.predict = T.function([self.input], self.layers[-1].get_output(self.input))
        self.compute_cost = T.function([self.input, self.target_output], self.cost)

    def get_all_params(self):
        return E.layers.get_all_params(self.layers[-1])

    def train(self):
        AbstractMethod()


class ModelSetting(object):
    def __init__(self, n_batches=None, learning_rate=None, n_time_steps=None, n_input_features=None,
                 n_output_features =None, cost_f=None, n_iterations=None, n_trains=None, serial=None):
        self._n_batches = n_batches
        self._learning_rate = learning_rate
        self._n_time_steps = n_time_steps
        self._n_input_features = n_input_features
        self._n_output_features = n_output_features
        self._cost_f = cost_f
        self._n_iterations = n_iterations
        # Number of transmitted variables
        self._n_trans = n_trains
        self._serial=serial


    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        assert isinstance(value, float)
        self._learning_rate = value

    @property
    def n_batches(self):
        return self._n_batches

    @n_batches.setter
    def n_batches(self, value):
        assert isinstance(value, int)
        self._n_batches = value

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @n_time_steps.setter
    def n_time_steps(self, value):
        assert isinstance(value, int)
        self._n_time_steps = value

    @property
    def n_input_features(self):
        return self._n_input_features

    @n_input_features.setter
    def n_input_features(self, value):
        assert isinstance(value, int)
        self._n_input_features = value

    @property
    def n_output_features(self):
        return self._n_output_features

    @n_output_features.setter
    def n_output_features(self, value):
        assert isinstance(value, int)
        self._n_output_features = value

    @property
    def cost_f(self):
        return self._cost_f

    @cost_f.setter
    def cost_f(self, value):
        assert hasattr(value, '__call__')
        self._cost_f = value

    @property
    def n_iterations(self):
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, value):
        assert isinstance(value, int)
        self._n_iterations = value

    @property
    def n_trans(self):
        return self._n_trans

    @n_trans.setter
    def n_trans(self, value):
        assert isinstance(value, int)
        self._n_trans = value

    @property
    def serial(self):
        return self._serial

    @serial.setter
    def serial(self, value):
        assert isinstance(value, E.serial.socket.SocketServer)
        self._serial = value


class LayerSetting(object):

    @staticmethod
    def iter_properties_of_class(cls):
        for varname in vars(cls):
            value = getattr(cls, varname)
            if isinstance(value, property):
                yield varname

    def properties(self):
        result = {}
        for cls in self.__class__.mro():
            for varname in self.iter_properties_of_class(cls):
                result[varname] = getattr(self, varname)
        return result

class InputLayerSetting(LayerSetting):
    def __init__(self, shape=None):
        super(InputLayerSetting, self).__init__()
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple), "The input is " + value + "However, we restrict input type to be int."
        self._shape = value



class LSTMLayerSetting(LayerSetting):
    def __init__(self, n_lstm_hidden_units=None):
        super(LSTMLayerSetting, self).__init__()
        self._n_lstm_hidden_units = n_lstm_hidden_units

    @property
    def num_units(self):
        return self._n_lstm_hidden_units

    @num_units.setter
    def num_units(self, value):
        assert isinstance(value, int)
        self._n_lstm_hidden_units = value



class ReshapeLayerSetting(LayerSetting):
    def __init__(self, reshape_shape=None):
        super(ReshapeLayerSetting, self).__init__()
        self._shape = reshape_shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple)
        self._shape = value

class DenseLayerSetting(LayerSetting):
    def __init__(self, dense_n_hidden_units=None, nonlineariry=None):
        super(DenseLayerSetting, self).__init__()
        self._dense_n_hidden_units = dense_n_hidden_units
        self._nonlinearity = nonlineariry

    @property
    def num_units(self):
        return self._dense_n_hidden_units

    @num_units.setter
    def num_units(self, value):
        assert isinstance(value, int)
        self._dense_n_hidden_units = value

    @property
    def nonlinearity(self):
        return self._nonlinearity

    @nonlinearity.setter
    def nonlinearity(self, value):
        self._nonlinearity = value

class ReshapeLayerSetting(LayerSetting):
    def __init__(self, reshape_to=None):
        super(ReshapeLayerSetting, self).__init__()
        self._shape = reshape_to

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple)
        self._shape = value


class Setting(ModelSetting):

    def __init__(self, n_batches=None, learning_rate=None, n_time_steps=None, n_input_features=None,
                 n_output_features =None, cost_f=None, n_iterations=None, n_trains=None, serial=None):
        super(Setting, self).__init__(n_batches=n_batches, learning_rate=learning_rate, n_time_steps=n_time_steps,
                                      n_input_features=n_input_features, n_output_features =n_output_features,
                                      cost_f=cost_f, n_iterations=n_iterations, n_trains=n_trains, serial=serial)
        self.layers = []
        self._previous_layer_n_out = None
        self.Layer = namedtuple('Layer', ['object', 'setting'], verbose=False)

    def l2s_map(self, layer):
        return {L.layers.InputLayer: InputLayerSetting,
                L.layers.ReshapeLayer: ReshapeLayerSetting,
                L.layers.LSTMLayer: LSTMLayerSetting,
                L.layers.DenseLayer: DenseLayerSetting}[layer]

    def append_layer(self, layer, layer_setting):
        for k, v in layer_setting.properties().items():
            if v is None:
                raise ValueError," The value of %s ->" % k + "is _None in layer" + layer.__name__
        self.layers.append(self.Layer(object=layer, setting=layer_setting))
        if layer in [L.layers.LSTMLayer, L.layers.DenseLayer]:
            self._previous_layer_n_out = layer_setting.num_units


    @property
    def previous_layer_n_out(self):
        if self._previous_layer_n_out == None:
            raise AttributeError,"Previous layer does not have Attribute num_units, may be previous layers is not a real layer?"
        return self._previous_layer_n_out
    @previous_layer_n_out.setter
    def previous_layer_n_out(self, value):
        self._previous_layer_n_out = value


    def get_layers_params(self):
        return [(layer, layer_setting.properties()) for layer, layer_setting in self.layers]





















