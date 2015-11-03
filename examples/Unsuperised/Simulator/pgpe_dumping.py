"""
This is another example of pgpe, this time I will use multiple layers.
It seems when I have more parameter, it is more difficult to get better result.
Although I didnt derive the back propagation part. However, this part affects performance for not
Worked fine
"""

import theano
import theano.tensor as T
import lasagne

import numpy as np
from numpy import array
from numpy import ones

import scipy.linalg as SLA
import scipy.signal as SIG
import scipy.io as sio


from einstein.tasks import EpisodicTask
from pybrain.rl.environments import Environment

import pygame

import matplotlib.pyplot as plt

import mdp

import einstein as e


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
        return [self.theta, self.thetadot, self.thetadotdot]

    def getAlpha(self): # numerical simulation of the ODE

        alpha = (-self.g / self.l)*np.sin(self.theta) - \
                (self.beta/self.m)*(self.thetadot) - \
                (self.l*self.gamma/self.m)*(self.thetadot**2)*np.sign(self.thetadot)
        return [alpha]

    def performAction(self, raw_actions):
        if self.keylock == False:
            if int(self.t/self.dt) > 500:
                self.act_force = float(raw_actions) * 10

                if self.act_force >7:
                    self.act_force = 7
                elif self.act_force < -7:
                    self.act_force = -7
            else:
                self.act_force = 0.0
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
        self.N = 10000
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
        self.screen.blit(self.render_text('reward       :'+self.format_number(self.cumreward,4),[255,255,255]),[1,464])

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


np.set_printoptions()

# Number of transmitted variables
N_TRANS = 5

# Input features
N_INPUT_FEATURES = 3

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


def theano_form(list, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param list: list to be transformed
    :param shape: output shape
    :return:
    """
    return array(list, dtype=theano.config.floatX).reshape(shape)


def test_iteration(task, all_params):
    """
    Give current value of weights, output all rewards
    :return:
    """
    _all_params = lasagne.layers.get_all_params(l_action_2_formed)
    _all_params[0].set_value(theano_form(all_params[0:N_HIDDEN], shape=(N_HIDDEN, 1)))
    _all_params[1].set_value(theano_form(all_params[N_HIDDEN::], shape=(N_INPUT_FEATURES, N_HIDDEN)))
    task.reset()
    while not task.isFinished():
        train_inputs = theano_form(task.getObservation(), shape=[N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES])
        actions = action_prediction(train_inputs)
        task.performAction(actions)



    coef = 50 # the bigger the more data
    thetas = task.env.thetas[::int(task.steps/coef)]
    thetas = array(thetas).reshape(len(thetas), 1)

    flow = get_flow2(thetas)

    cmreward = flow(thetas)

    if cmreward[0] - cmreward[1] < 0:
        cmreward = -cmreward

    plt.figure(1)
    plt.plot(cmreward)
    plt.savefig("figure1")

    return flow

def one_iteration(task, all_params, flow):
    """
    Give current value of weights, output all rewards
    :return:
    """
    _all_params = lasagne.layers.get_all_params(l_action_2_formed)
    _all_params[0].set_value(theano_form(all_params[0:N_HIDDEN], shape=(N_HIDDEN, 1)))
    _all_params[1].set_value(theano_form(all_params[N_HIDDEN::], shape=(N_INPUT_FEATURES, N_HIDDEN)))
    task.reset()

    for theta in np.linspace(-np.pi/2, np.pi, 10):
        while not task.isFinished():
            train_inputs = theano_form(task.getObservation(), shape=[N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES])
            actions = action_prediction(train_inputs)
            task.performAction(actions)



        coef = 35
        thetas = task.env.thetas[::int(task.steps/coef)]
        thetas = array(thetas).reshape(len(thetas), 1)
        cmreward = flow(thetas)
        cmreward += min(cmreward)



    return cmreward

def sample_parameter(sigma_list):
    """
    sigma_list contains sigma for each parameters
    """
    return np.random.normal(0., sigma_list)

def extract_parameter(params):
    current = array([])
    for param in params:
        current = np.concatenate((current, param.get_value().flatten()), axis=0)
    return current


def get_flow(thetas):
    flow = (mdp.nodes.PolynomialExpansionNode(2) +
            mdp.nodes.WhiteningNode(svd=True, reduce=True) +
            mdp.nodes.SFANode(output_dim=1))
    flow.train(thetas)
    return flow

def get_flow2(thetas):
    flow = (mdp.nodes.EtaComputerNode() +
            mdp.nodes.TimeFramesNode(2) +
            mdp.nodes.PolynomialExpansionNode(2) +
            mdp.nodes.SFANode(output_dim=1) +
            mdp.nodes.EtaComputerNode() )
    flow.train(thetas)
    return flow

def sfa(thetas):

    def sort_eig(D,E):
        idx = D.argsort()[::-1]
        D = D[idx]
        E = E[:,idx]
        return D,E

    thetadots = np.diff(thetas, n=1, axis=0)
    data = array([thetas[1::], thetadots])
    dm = array([data[0, :],
                   data[1, :],
                   data[0, :]**2,
                   data[0, :] * data[1, :],
                   data[1, :]**2,
                   ])
    dm -= np.mean(dm,axis=1).reshape(5,1)

    dm = dm.T

    C = np.dot(dm.T, dm)/data[0,:].shape[0]

    dw= np.dot(dm, SLA.inv(SLA.sqrtm(C)))

    dslow = np.zeros((dw.shape[0]-1, dw.shape[1]))
    for i in xrange(5):
        dslow[:, i]= SIG.convolve(dw[:,i], array([1, 1])/2.,'valid')
    C3 = np.dot(dslow.T, dslow) /len(thetas)
    D3, E3 = sort_eig(*np.linalg.eig(C3))

    sfa = np.dot(dw, E3)



    sio.savemat("params.mat",{#"forces" : self.act_forces,
                              #"x_positions" : self.x_positions,
                              #"y_positions" : self.y_positions,
                              "thetas": thetas})
                              #"thetadots": self.thetadots,
                              #"thetadotdots":self.thetadotdots})
    #raw_input()
    return sfa

if __name__ == "__main__":
    # Construct vanilla RNN: One recurrent layer (with input weights) and one
    # dense output layer
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


    env = IncEnergyEnvironment(theta=0.4,
                               m = 0.3010,
                               l =0.5,
                               g=9.8,
                               beta=0.20,
                               gamma=0.20,
                               dt=0.001,
                               steps_per_update=100)
    task = IncEnergyTask(env)



    baseline = None
    num_parameters = N_HIDDEN + N_HIDDEN * N_INPUT_FEATURES # five parameters
    epsilon = 1 # initial number sigma
    sigma_list = ones(num_parameters) * epsilon
    deltas = sample_parameter(sigma_list=sigma_list)
    best_reward = -1000

    current = extract_parameter(params=all_params)
    arg_reward = []


    flow = test_iteration(task = task, all_params = np.zeros(current.shape))


    #XT.grade(all_params)
    for n in xrange(1000):

         # current parameters
        deltas = sample_parameter(sigma_list=sigma_list)
        reward1 = sum(one_iteration(task=task, all_params=current + deltas, flow=flow))
        if reward1 > best_reward:
            best_reward = reward1
        reward2 = sum(one_iteration(task= task, all_params=current - deltas, flow=flow))
        if reward2 > best_reward:
            best_reward = reward2
        mreward = (reward1 + reward2) / 2.

        if baseline is None:
            # first learning step
            baseline = mreward
            fakt = 0.
            fakt2 = 0.
        else:
            #calc the gradients
            if reward1 != reward2:
                #gradient estimate alla SPSA but with likelihood gradient and normalization
                fakt = (reward1 - reward2) / (2. * best_reward - reward1 - reward2)
            else:
                fakt=0.
            #normalized sigma gradient with moving average baseline
            norm = (best_reward - baseline)
            if norm != 0.0:
                fakt2=(mreward-baseline)/(best_reward-baseline)
            else:
                fakt2 = 0.0
        #update baseline
        baseline = 0.99 * (0.9 * baseline + 0.1 * mreward)


        # update parameters and sigmas
        current = current + LEARNING_RATE * fakt * deltas

        if fakt2 > 0.: #for sigma adaption alg. follows only positive gradients
            #apply sigma update locally
            sigma_list = sigma_list + LEARNING_RATE * fakt2 * (deltas * deltas - sigma_list * sigma_list) / sigma_list


        arg_reward.append(mreward)
        if not n%100:
            print baseline
            print "best reward", best_reward, "average reward", sum(arg_reward)/len(arg_reward)
            arg_reward = []
            e.fileio.save_to_pickle_file(current,"damping_current%d"%n)
            e.fileio.save_to_pickle_file(sigma_list,"damping_sigma_list%d"%n)
