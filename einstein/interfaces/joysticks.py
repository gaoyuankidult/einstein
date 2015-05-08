__author__ = 'gao'
import pygame
from threading import Timer




class Joystick(object):
    def __init__(self):
        pygame.init()
        pygame.joystick.init()  # initialize joystick system
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

    def __del__(self):
        pygame.quit()


class FormulaForceEx(Joystick):
    def __init__(self):
        super(FormulaForceEx, self).__init__()
        assert len(self.joysticks) == 1
        self.ex = self.joysticks[0]
        self.ex.init()  # initialize the wheel

    def show_wheel_value(self):
        # EVENT PROCESSING STEP
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                pass
        axis_value_0 = self.ex.get_axis(0)
        print "Axis {} value: {:>6.3f}".format(0, axis_value_0)

    def read_wheel_value(self, times):
        """
        :param times: indicate how many time will wheel values be read per second
        :return:
        """
        self.t = Timer(1./times, self.show_wheel_value)
        self.t.start()
        raw_input()

    def __del__(self):
        super(FormulaForceEx, self).__del__()
        self.t.join()

