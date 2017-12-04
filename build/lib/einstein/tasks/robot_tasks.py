from pybrain.rl.environments import EpisodicTask
from numpy import clip

class BaxterReachTask(EpisodicTask):
    """ The task of balancing some pole(s) on a cart """
    def __init__(self, env=None, maxsteps=1000, desiredValue = 0, tolorance = 0.3):
        """
        :key env: (optional) an instance of a CartPoleEnvironment (or a subclass thereof)
        :key maxsteps: maximal number of steps (default: 1000)
        """
        self.desiredValue = desiredValue
        EpisodicTask.__init__(self, env)
        self.N = maxsteps
        self.t = 0
        self.tolorance = tolorance


        # self.sensor_limits = [None] * 4
        # actor between -10 and 10 Newton
        self.actor_limits = [(-15, 15)]

    def reset(self):
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, actions):
        self.t += 1
        actions = clip(actions, -0.05, 0.05) # set speed limit
        EpisodicTask.performAction(self, actions)

    def isFinished(self):

        # Neutral place
        # x=0.6359848748431522, y=0.8278984542845692, z=0.19031037139621507
        end_effector_pose = self.env.getEndEffectorPosition()
        if end_effector_pose.x - 0.6359848748431522 < self.tolorance and \
            end_effector_pose.y - 0.8278984542845692< self.tolorance and \
            end_effector_pose.z - 0.19031037139621507 < self.tolorance:
            return True
        elif self.t == self.N:
            return True
        return False

    def getReward(self):
        end_effector_pose = self.env.getEndEffectorPosition()
        if end_effector_pose.x - 0.6359848748431522 < self.tolorance and \
            end_effector_pose.y - 0.8278984542845692< self.tolorance and \
            end_effector_pose.z - 0.19031037139621507 < self.tolorance:
            reward = (self.N - self.t)
        else:
            reward = -1
        return reward

    def setMaxLength(self, n):
        self.N = n