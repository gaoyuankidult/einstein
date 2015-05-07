"""
This file contains the manipulation of model's information
"""
import einstein as E
class ModelRecorder(object):
    def __init__(self):
        self.clear()

    def record(self, value):
        self.records[self.t].append(value)
        assert E.tools.check_list_depth(self.records) == 2, \
            "The records should be two dimensional array, maybe forget using ModelRecorder.new_epoch() ?"

    def clear(self):
        # all records
        self.records = []
        # current epoch number
        self.t = -1

    def record_real_tests(self):
        pass

    def new_epoch(self):
        self.records.append([])
        self.t += 1

    def print_running_avg(self, current_epoch, n_real_example, interval, steps):
        """
        Print running average
        :param current_epoch:
        :param n_real_example:
        :param interval:
        :param steps:
        :return: average value
        """
        assert isinstance(steps, int)
        assert isinstance(interval, int)
        mean_value = E.tools.mean(self.records[self.t][-steps::1])
        if current_epoch%interval == 0:
            print "Current epoch %i, real example %d, Interval %d, steps %d, average reward %f" \
                  % (current_epoch,
                     n_real_example,
                     interval,
                     steps,
                     mean_value)
        return mean_value
    def print_real_and_sim_diff(self, real_experiment, sim_experiment, params):

        real_reward = real_experiment.one_epicode(params)

        sim_reward = sim_experiment.one_epicode(params)

        print "For the same parameters, the real reward of the system is %f, the sim reward of system is %f" % \
              (real_reward, sim_reward)