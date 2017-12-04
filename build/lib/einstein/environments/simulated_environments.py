from numpy import concatenate

import einstein as E

from pybrain.rl.environments.graphical import GraphicalEnvironment


from numpy import array

from scipy import random


class SimCartPoleEnvironment(GraphicalEnvironment):
    randomInitialization = True
    outdim = 4
    def __init__(self, critic_model):
        super(SimCartPoleEnvironment, self).__init__()
        self.critic_model = critic_model
        self.predict = self.critic_model.predict
        self.reset()

    def performAction(self, action):
        self.actions_sequence.append(action[0][0])
        predict_input = concatenate([E.tools.theano_form(self.actions_sequence.data,
                                                         shape=(self.critic_model.setting.n_batches,
                                                                self.critic_model.setting.n_time_steps, 1)),
                                     E.tools.theano_form(self.sensors_sequence.data,
                                                         shape=(self.critic_model.setting.n_batches,
                                                                self.critic_model.setting.n_time_steps, 4))], axis=2)

        prediction = self.predict(predict_input)
        self.sensors = prediction[0][-1][:]
        #print
        #print self.actions_sequence.data
        #print self.sensors_sequence.data
        #raw_input()
        self.sensors_sequence.append(self.sensors)

    def getSensors(self):
        #print self.sensors
        return array(self.sensors)

    def getPoleAngles(self):
        return [self.sensors[0]]

    def getCartPosition(self):
        return self.sensors[2]

    def reset(self):
        if self.randomInitialization:
            angle = random.uniform(-0.2, 0.2)
            pos = random.uniform(-0.5, 0.5)
        else:
            angle = -0.2
            pos = 0.2
        self.sensors_sequence = E.tools.RingBuffer(self.critic_model.setting.n_time_steps, ivalue=[0.0] * self.outdim)
        self.actions_sequence = E.tools.RingBuffer(self.critic_model.setting.n_time_steps, ivalue=[0.0])
        self.sensors = [angle, 0.0, pos, 0.0]
        self.sensors_sequence.append(self.sensors)


