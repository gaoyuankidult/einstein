import time,math,os
import numpy as np
import matplotlib.pyplot as plt
import collections as co
from einstein.fileio import FileIO
import scipy.io as sio

global useDisplay # so that other functions can access it
useDisplay = True # set to False if you don't have pygame
if useDisplay:
    import pygame
    pygame.init()
    pygame.mixer.init()

    # For the readout on the display
    global font
    font = pygame.font.SysFont('lucida console',12)

    global clock
    clock=pygame.time.Clock()

# Used to save code in the render function
def renderText(text,col):
    return font.render(text,True,col)
def formatNumber(number,rnd):
    st = ''
    if sign(number)!=-1:st += ' '
    st += str(round(number,rnd))
    return st
def sign(x):
    if x==0:return x
    return abs(x)/x
def floatIsEqual(f1,f2,gamma=1e-5):
    return (2*(abs(f1-f2)/(abs(f1)+abs(f2))))<gamma


class Plotter(object):
    def __init__(self):
        self.fig = plt.figure(1)
        plt.ion()
        plt.show(False)
        plt.draw()
        self.p1q = co.deque([0]*10,maxlen=10)
        self.p2q = co.deque([0]*10,maxlen=10)

    def plot(self):
        self.plot1()

    def plot1(self):
        fig, ax = plt.subplot(211)
        background = fig.canvas.copy_from_bbox(ax.bbox)
        fig.canvas.restore_region(background)

        # redraw just the points
        ax.draw_artist(points)

        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)


        plt.scatter(xrange(len(list(self.p1q))),list(self.p1q))
        plt.draw()

class pendulumSim(object):
    def __init__(self,theta,m,l,g,beta,gamma,step,stepsPerUpdate,noDisplay):
        #static variables
        self.thetaNaught = theta # initial angular displacement
        self.m=m # mass of the bob
        self.l=l # length of string
        self.g=g # gravitational acceleration
        self.beta=beta     # linear drag
        self.gamma = gamma # quadratic drag
        self.step=step     # amount of time per step in the differential.
                           # lower values take more time but are more accurate.
        self.stepsPerUpdate = stepsPerUpdate # used to sync display to simulation with a sane framerate
                                             # defaults to 1 second per second
        self.noDisplay = (noDisplay or not useDisplay) # used to override the global setting if useDisplay==True

        # control parameters
        self.control_cnt = 0
        self.act_force = 0
        self.thetadotdot = 0
        self.act = 0 # 0: push left, 1: push right, 2: drag

        # recording
        self.x_positions = []
        self.y_positions = []
        self.thetas = []
        self.thetadots = []
        self.thetadotdots = []
        self.time_array = []
        self.act_forces = []
        self.breaking_sig = []

        # plotting function
        self.plotter = Plotter()

        #dynamic variables
        self.theta = theta # angular position of pendulum
        self.thetadot = 0  # current angular velocity
        self.t = 0         # current time

        #display
        if not self.noDisplay:
            self.screen=pygame.display.set_mode((512,512))

        #period
        self.isStopped = True

    def getAlpha(self): # numerical simulation of the ODE
        alpha = (-self.g / self.l)*math.sin(self.theta) - \
                (self.beta/self.m)*(self.thetadot) - \
                (self.l*self.gamma/self.m)*(self.thetadot**2)*sign(self.thetadot)


        return alpha

    def makeact(self):
        def push_left():
            self.act = 0
            return -abs(np.random.randn(1)) -4
        def push_right():
            self.act = 1
            return abs(np.random.randn(1)) + 4
        opt = np.random.randint(0, 2)
        return [push_left, push_right,][opt]()

    def doStep(self,dt, # performs a step of lenght dt
               scale_beta = 0.0 # for stopping process
               ):

        if self.control_cnt != 300 :
            self.thetadotdot =self.getAlpha() + self.act_force
            self.control_cnt += 1
        else:
            self.act_force = self.makeact()
            self.thetadotdot = self.getAlpha() + self.act_force
            self.control_cnt = 0


        if scale_beta != 0:
            self.thetadotdot -= self.act_force
            self.thetadotdot += (self.beta/self.m)*(self.thetadot)
            self.thetadotdot -= (scale_beta/self.m)*(self.thetadot)

        self.thetadot += self.thetadotdot*dt
        self.theta += self.thetadot*dt
        self.t += dt

        self.act_forces.append(self.act_force)
        self.x_positions.append(self.l*math.sin(self.theta))
        self.y_positions.append(self.l*math.cos(self.theta))
        self.thetas.append(float(self.theta))
        self.thetadots.append(float(self.thetadot))
        self.thetadotdots.append(float(self.thetadotdot))
        self.time_array.append(self.t)
        self.breaking_sig.append(scale_beta)

    def display(self): # updates the display
        self.screen.fill((0,0,0))
        pygame.draw.line(self.screen,[255,255,255],[256,192],[256+int(256*self.l*math.sin(self.theta)),192+int(256*self.l*math.cos(self.theta))])
        pygame.draw.circle(self.screen,[255,255,255],[256+int(256*self.l*math.sin(self.theta)),192+int(256*self.l*math.cos(self.theta))],4)



        self.screen.blit(renderText('theta      :'+formatNumber(self.theta,6),[255,255,255]),[1,384])
        self.screen.blit(renderText('thetadot   :'+formatNumber(self.thetadot,6),[255,255,255]),[1,400])
        self.screen.blit(renderText('thetadotdot:'+formatNumber(self.thetadotdot,6),[255,255,255]),[1,416])
        self.screen.blit(renderText('time       :'+formatNumber(self.t,4),[255,255,255]),[1,432])



        #self.plotter.p1q.append(self.theta)
        #self.plotter.p2q.append(self.thetadot)
        #self.plotter.plot()

        pygame.display.update()

    def getIsStopped(self):
        if abs(self.t) > 30:
            self.isStopped = True
            return True
        else:
            self.isStopped=False
            return False

    def main(self): # eschew first; get avg of first 10 since that's what I did in trials
        self.times = []
        self.stepcount = 0
        scale_beta = 0
        while not self.getIsStopped():
            events = pygame.event.get() # to prevent freezing
            if not self.noDisplay:
                if self.stepcount == self.stepsPerUpdate:
                    self.display();self.stepcount=0
                    clock.tick(int(1/(self.step*self.stepsPerUpdate)))
                else:
                    self.stepcount+=1
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        scale_beta = 0.9
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        scale_beta = 0
            self.doStep(self.step, scale_beta)
        save_file = FileIO()
        save_file.save_pickle(self.act_forces, "forces")
        save_file.save_pickle(self.x_positions, "x_positions")
        save_file.save_pickle(self.y_positions, "y_positions")
        save_file.save_pickle(self.thetas, "thetas")
        save_file.save_pickle(self.thetadots, "thetasdots")
        save_file.save_pickle(self.thetadotdots, "thetasdotdots")
        save_file.save_pickle(self.time_array, "times")


        sio.savemat("params.mat",{#"forces" : self.act_forces,
                                  #"x_positions" : self.x_positions,
                                  #"y_positions" : self.y_positions,
                                  "thetas": self.thetas,
                                  "breaking_sig":self.breaking_sig})
                                  #"thetadots": self.thetadots,
                                  #"thetadotdots":self.thetadotdots})

        pygame.display.quit()

        return None



### What units I assumed
### mass    : kg
### length  : meters
### beta    : kg/s
### gamma   : kg/m
### time    : seconds



# Retreieves the period given input parameters.
# beta      = coefficient of linear drag (kg/s)
# gamma     = coefficient of quadratic drag (kg/m)
# step      = length of time interval for simulation.
#             I don't recommend values higher than 0.001
# numSwings = number of swings to average the period over
def getPeriod(mass,length,degrees,beta,gamma,step=0.0001,g=9.8,numSwings=2):
    rad = 2*math.pi*degrees/360.0 # degrees -> radians
    numSym=pendulumSim(rad,mass,length,g,beta,gamma,step,1,True) # doesn't do display
    return numSym.main(numSwings+1)

# for a visual of the pendulum. Good for reality checks and sanity.
def doPendulumDisplay(mass,length,degrees,beta,gamma,step=0.0001,spf=100,g=9.8):
    if useDisplay:
        rad = 2*math.pi*degrees/360.0 # degrees -> radians
        numSym=pendulumSim(rad, mass, length, g , beta, gamma, step, spf,False)
        numSym.main() # non-terminating, I swear
    else:print("Display not enabled. Please enable display to use this function.")

displayExample=True
if displayExample:
    doPendulumDisplay(mass=0.3010,
                      length=0.5,
                      degrees=0,
                      beta=0.01,
                      gamma=0.01)

# gets a set of period measures corresponding to a list of initial angular displacements
def getPeriodMeasure(degreeList,mass,length,beta,gamma,step=0.0001,g=9.8):
    return [round(getPeriod(mass,length,deg,beta,gamma,step,g),8) for deg in degreeList]
