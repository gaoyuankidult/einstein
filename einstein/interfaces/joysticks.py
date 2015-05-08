__author__ = 'gao'
import pygame
import einstein as E
from threading import Timer
from threading import Lock


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
        self.t = None  # thread
        self.wheel_value = None
        self.lock = Lock()

    def update_wheel_value(self, times=None):
        """
        Indicates how many times will function show in 1 second
        :param times:
        :return:
        """
        if times is not None:
            self.t = Timer(1./times, self.update_wheel_value, [times]).start()
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                pass
        with self.lock:
            self.wheel_value = self.ex.get_axis(0)
        print "Axis {} value: {:>6.3f}".format(0, self.wheel_value)

    def terminate_wheel_monitor(self):
        E.tools.check_none_join(self.t)

    def get_wheel_value(self):
        with self.lock:
            self.wheel_value = self.ex.get_axis(0)
        return self.wheel_value

    def __del__(self):
        super(FormulaForceEx, self).__del__()
        self.terminate_wheel_monitor()



