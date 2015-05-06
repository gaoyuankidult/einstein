"""
This file contains extend Limb with objective check. i.e. check whether the limb has reach the objective state or not
"""
from baxter_interface import Limb
from baxter_interface import settings
import baxter_dataflow

class ELimb(Limb):
    def __init__(self, limb):
        super(ELimb, self).__init__(limb)

    def check_objective_position(self, target, threshold=settings.JOINT_ANGLE_TOLERANCE):
        print self.joint_angles()



