"""
This file contains different operation on layers
"""
import einstein as E


def train_critic_network(network_model, training_input, training_output,  training_method=None, training_step=None, training_percent=None):
    assert (training_percent is None or training_step is None)
    all_data = zip(training_input, training_output)
    if training_step is None:
        training_step = int(training_percent * len(all_data))
    costs = []
    if training_method == "direct training":
        for input, output in all_data:
            critic_train_input = E.tools.theano_form(input, shape=network_model.get_input_shape())
            critic_train_output = E.tools.theano_form(output, shape=network_model.get_output_shape())
            costs.append(network_model.train(critic_train_input, critic_train_output))
    elif training_method == "stochastic training":
        for _ in xrange(training_step):
            i = E.tools.random.choice(range(len(all_data)))
            input, output = all_data[i]
            critic_train_input = E.tools.theano_form(input, shape=network_model.get_input_shape())
            critic_train_output = E.tools.theano_form(output, shape=network_model.get_output_shape())
            costs.append(network_model.train(critic_train_input, critic_train_output))
    return costs


def set_network_params(network_model, file_name):
    params = E.tools.load_pickle_file(file_name)
    for param, layer in zip(params, network_model.layers):
        if param == []:
            pass
        else:
            # get each parameter of the layer and set values to all parameters
            _all_params = layer.get_params()
            for num_w, w in enumerate(param):
                _all_params[num_w].set_value(w.get_value())


if __name__ == "__main__":
    file_name = "/home/deepthree/Desktop/deepcontrol_alpha/data/model_nu_data/0-199/trained_nu_model_params"