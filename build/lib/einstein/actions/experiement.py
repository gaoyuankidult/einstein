import einstein as E


class Experiment(object):
    def __init__(self, task, actor_model):
        self.task = task
        self.actor_model = actor_model

    def reset(self, all_params):

        self.rewards = []
        self.observations = []
        self.actions = []
        _all_params = self.actor_model.get_all_params()
        _all_params[0].set_value(E.tools.theano_form(all_params, shape=(4, 1)))
        self.task.reset()

    def one_epicode(self, all_params):
        """
        Give current value of weights, output all rewards
        :return:
        """
        self.reset(all_params)
        while not self.task.isFinished():
            obs = self.task.getObservation()
            self.observations.append(obs)
            states = E.tools.theano_form(obs, shape=[self.actor_model.setting.n_batches,
                                                     1,
                                                     self.actor_model.setting.n_input_features]) # this is for each time step
            model_action_result = self.actor_model.predict(states)
            self.actions.append(model_action_result.reshape(1))
            self.task.performAction(model_action_result)
            self.rewards.append(self.task.getReward())
        self.last_obs = self.task.getObservation()
        return sum(self.rewards)


class ThoughtExperiment(Experiment):
    def __init__(self, task, actor_model, critic_model):
        super(ThoughtExperiment, self).__init__(task=task,
                                                actor_model=actor_model)
        self.critic_model = critic_model

    def one_epicode(self, all_params):
        self.reset(all_params)
        while not self.task.isFinished():
            #print self.task.env.getPoleAngles(), self.task.env.getCartPosition()
            obs = self.task.getObservation()
            self.observations.append(obs)
            states = E.tools.theano_form(obs, shape=[self.actor_model.setting.n_batches,
                                                     1,
                                                     self.actor_model.setting.n_input_features]) # this is for each time step
            model_action_result = self.actor_model.predict(states)
            self.actions.append(model_action_result.reshape(1))
            self.task.performAction(model_action_result)
            self.rewards.append(self.task.getReward())
        self.last_obs = self.task.getObservation()
        return sum(self.rewards)



class RealExperiment(Experiment):
    def __init__(self, task, actor_model):
        super(RealExperiment, self).__init__(task=task,
                                             actor_model=actor_model)

    def get_training_data(self, unfolding=1):
        self.actions = E.tools.theano_form(self.actions, shape=(len(self.actions), 1))
        self.observations = E.tools.theano_form(self.observations, shape=(len(self.observations), 4))
        predicted_obs = E.tools.concatenate([self.observations[1::], [self.last_obs]])
        input_data = E.tools.concatenate([self.actions, self.observations], axis=1)
        output_data = predicted_obs
        if unfolding >= len(input_data):
            return [input_data], [output_data]
        else:
            critic_train_inputs = list(E.tools.make_chunks(input_data, unfolding))
            critic_train_outputs = list(E.tools.make_chunks(output_data, unfolding))
            return critic_train_inputs, critic_train_outputs