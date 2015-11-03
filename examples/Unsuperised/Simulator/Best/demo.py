import einstein as e

import lasagne

import theano

import theano.tensor as T

import pygame

import numpy as np

from einstein.tasks import EpisodicTask
from pybrain.rl.environments import Environment

import code

import matplotlib.pyplot as plt

fileio = e.fileio.FileIO()
damping_current = fileio.load_pickle("damping_best")
driving_current = fileio.load_pickle("driving_best")
zeros = np.zeros(*damping_current.shape)


print("damping params", damping_current)
print("driving params", driving_current)
num_points = 10
params = zip(*[np.linspace(i,j,num_points) for i,j in zip(damping_current,driving_current)])

# Number of transmitted variables
N_TRANS = 5

# Input features
N_INPUT_FEATURES = 2

# Output Features
N_ACTIONS = 1

# Length of each input sequence of data
N_TIME_STEPS = 1  # in cart pole balancing case, x, x_dot, theta, theta_dot and reward are inputs


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 5

# This means how many sequences you would like to input to the sequence.
N_BATCH = 1

# SGD learning rate
LEARNING_RATE = 2e-1

# Number of iterations to train the net
N_ITERATIONS = 1000000

# Forget rate
FORGET_RATE = 0.9

# Number of reward output
N_REWARD = 1

GRADIENT_METHOD = 'sgd'


l_in = lasagne.layers.InputLayer(shape=(N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES))
# Followed by a Dense Layer to Produce Action
l_action_1 = lasagne.layers.DenseLayer(incoming=l_in,
                                       num_units=N_HIDDEN,
                                       nonlinearity=None,
                                       b=None)

l_action_1_formed = lasagne.layers.ReshapeLayer(incoming=l_action_1,
                                                shape=(N_BATCH, N_TIME_STEPS, N_HIDDEN))

l_action_2 = lasagne.layers.DenseLayer(incoming=l_action_1_formed,
                                       num_units=N_ACTIONS,
                                       nonlinearity=None,
                                       b=None)

l_action_2_formed = lasagne.layers.ReshapeLayer(incoming=l_action_2,
                                                shape=(N_BATCH, N_TIME_STEPS, N_ACTIONS))
# Cost function is mean squared error
input = T.tensor3('input')
target_output = T.tensor3('target_output')
action_prediction = theano.function([input], l_action_2_formed.get_output(input))
all_params = lasagne.layers.get_all_params(l_action_2_formed)

def theano_form(list, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param list: list to be transformed
    :param shape: output shape
    :return:
    """
    return np.array(list, dtype=theano.config.floatX).reshape(shape)

def test_iteration(task, all_params_set):
    """
    Give current value of weights, output all rewards
    :return:
    """
    param_no = 0
    all_params = params[param_no]
    _all_params = lasagne.layers.get_all_params(l_action_2_formed)
    _all_params[0].set_value(theano_form(all_params[0:N_HIDDEN], shape=(N_HIDDEN, 1)))
    _all_params[1].set_value(theano_form(all_params[N_HIDDEN::], shape=(N_INPUT_FEATURES, N_HIDDEN)))
    task.reset()
    while not task.isFinished():
        train_inputs = theano_form(task.getObservation(), shape=[N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES])
        actions = action_prediction(train_inputs)
        task.performAction(actions)
        task.screen.blit(task.render_text('current params set       :'+  str(param_no), [255,255,255]),[1,464])
        pygame.display.update()


        events = pygame.event.get() # to prevent freezing
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    param_no += 1
                    if param_no >= len(params) -1:
                        param_no = len(params) -1
                elif event.key == pygame.K_DOWN:
                    param_no -= 1
                    if param_no <= 0:
                        param_no = 0

        all_params = params[param_no]
        _all_params[0].set_value(theano_form(all_params[0:N_HIDDEN], shape=(N_HIDDEN, 1)))
        _all_params[1].set_value(theano_form(all_params[N_HIDDEN::], shape=(N_INPUT_FEATURES, N_HIDDEN)))

    thetas = task.env.thetas
    forces = task.env.forces


    return thetas, forces

class IncEnergyEnvironment(Environment):
    def __init__(self, theta, m, l, g, beta, gamma, dt, steps_per_update):
        Environment.__init__(self)
        self.m=m # mass of the bob
        self.l=l # length of string
        self.g=g # gravitational acceleration

        self.beta = beta     # linear drag
        self.gamma = gamma # quadratic drag
        self.dt = dt     # amount of time per step in the differential.
                           # lower values take more time but are more accurate.
        self.steps_per_update = steps_per_update # used to sync display to simulation with a sane framerate
                                             # defaults to 1 second per second

        self.act_force = 0.0

        self.t = 0
        self.theta_origin = theta
        self.theta = 0.0
        self.thetadot = 0.0
        self.thetadotdot = 0.0

        self.thetas = []
        self.energies = []
        self.forces = []

        self.keylock = False

    def getSensors(self):
        return [self.theta, self.thetadot]

    def getAlpha(self): # numerical simulation of the ODE

        alpha = (-self.g / self.l)*np.sin(self.theta) - \
                (self.beta/self.m)*(self.thetadot) - \
                (self.l*self.gamma/self.m)*(self.thetadot**2)*np.sign(self.thetadot)
        return [alpha]

    def performAction(self, raw_actions):
        if self.keylock == False:
            self.act_force = float(raw_actions) * 10
            if self.act_force >7:
                self.act_force = 7
            elif self.act_force < -7:
                self.act_force = -7

        self.step()


    def human_control(self):
        events = pygame.event.get() # to prevent freezing
        for event in events:
            if event.type == pygame.KEYDOWN:
                self.keylock = True
                if event.key == pygame.K_LEFT:
                    self.act_force = -7
                elif event.key == pygame.K_RIGHT:
                    self.act_force = 7

            if event.type == pygame.KEYUP:
                self.keylock = False
                if event.key == pygame.K_LEFT:
                    self.act_force = 0
                elif event.key == pygame.K_RIGHT:
                    self.act_force = 0

    def step(self):
        self.thetadotdot = self.getAlpha()[0] + self.act_force
        self.control_cnt = 0


        self.thetadot += self.thetadotdot*self.dt
        self.theta += self.thetadot*self.dt
        self.t += self.dt

        self.thetas.append(self.theta)
        self.forces.append(self.act_force)

        #potential_energy = self.l*np.cos(self.theta) * self.m * self.g
        #kinetic_energy = self.m * (self.thetadot * self.l) ** 2 / 2
        #self.energies.append(potential_energy + kinetic_energy)

    def reset(self):
        self.theta = self.theta_origin
        self.thetadot = 0
        self.thetadotdot = 0
        self.t = 0.0

        self.thetas = []
        self.energies = []

        self.act_force = 0.0

class IncEnergyTask(EpisodicTask):
    """ The task of balancing some pole(s) on a cart """
    def __init__(self, env=None):
        self.N = 100000
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((512,512))
        self.step_count = 0

        self.font = pygame.font.SysFont('lucida console',12)
        self.clock= pygame.time.Clock()

        super(IncEnergyTask, self).__init__(env)

    def reset(self):
        super(IncEnergyTask, self).reset()
        self.steps = 0

    def performAction(self, action):
        if self.step_count == self.env.steps_per_update:
            self.display();
            self.stepcount=0
        else:
            self.step_count+=1

        self.steps += 1

        super(IncEnergyTask, self).performAction(action)

    def render_text(self, text, col):
        return self.font.render(text,True,col)

    def format_number(self, number, rnd):
        st = ''
        if np.sign(number)!=-1:st += ' '
        st += str(round(number,rnd))
        return st

    def display(self): # updates the display
        self.screen.fill((0,0,0))
        pygame.draw.line(self.screen,[255,255,255],[256,192],[256+int(256*self.env.l*np.sin(self.env.theta)),192+int(256*self.env.l*np.cos(self.env.theta))])
        pygame.draw.circle(self.screen,[255,255,255],[256+int(256*self.env.l*np.sin(self.env.theta)),192+int(256*self.env.l*np.cos(self.env.theta))],4)

        self.screen.blit(self.render_text('theta      :'+self.format_number(self.env.theta,6),[255,255,255]),[1,384])
        self.screen.blit(self.render_text('thetadot   :'+self.format_number(self.env.thetadot,6),[255,255,255]),[1,400])
        self.screen.blit(self.render_text('thetadotdot:'+self.format_number(self.env.thetadotdot,6),[255,255,255]),[1,416])
        self.screen.blit(self.render_text('time       :'+self.format_number(self.env.t,4),[255,255,255]),[1,432])
        if self.env.act_force < 0:
            self.screen.blit(self.render_text('force: left       :'+self.format_number(self.env.act_force,4),[255,255,255]),[1,448])
        else:
            self.screen.blit(self.render_text('force: right       :'+self.format_number(self.env.act_force,4),[255,255,255]),[1,448])


        pygame.display.update()

    def getReward(self):
        angle = map(abs, self.env.getSensors())[0]
        if angle >= np.pi/2:
            reward = (self.N - self.steps)/4.
        else:
            reward = angle ** 2
        return reward

    def isFinished(self):
        if self.steps >= self.N:
            return True
        return False

env = IncEnergyEnvironment(theta=0.4,
                           m = 0.3010,
                           l =0.5,
                           g=9.8,
                           beta=0.20,
                           gamma=0.20,
                           dt=0.001,
                           steps_per_update=100)
task = IncEnergyTask(env)


thetas, forces = test_iteration(task=task, all_params_set=params)
plt.figure(1)
plt.plot(thetas, forces)
plt.figure(2)
plt.grid()
plt.plot(thetas, label="thetas")
plt.plot(forces, label="forces")
plt.legend()
plt.savefig("thetas_forces")