import einstein as E
import rospy
from pybrain.rl.environments import Environment
from pybrain.rl.environments import EpisodicTask


class BaxterEnvironment(Environment):
    """
    This class takes care of communicating with simulator
    """
    def __init__(self, baxter_interface):
        Environment.__init__(self)
        print("Initializing node... ")
        rospy.init_node("test")
        self.rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        init_state = self.rs.state().enabled

        def clean_shutdown():
            print("\nExiting example...")
            if not init_state:
                print("Disabling robot...")
                self.rs.disable()
        rospy.on_shutdown(clean_shutdown)

        print("Enabling robot... ")
        self.rs.enable()

        self.right_arm = baxter_interface.Limb('left')
        self.joint_names = self.right_arm.joint_names()


    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return
            vector has 4 elements: theta, theta', s, s' (s being the distance from the
            origin).
        """
        return self.right_arm.joint_angles().values() # get reversed order states from wrist to shoulder

    def performAction(self, raw_actions):
        """

        :param action: action is a list of lens 7.
        :return:
        """
        actions = dict(zip(self.right_arm.joint_names(),
                          raw_actions[0].flatten().tolist()))
        self.actions = actions
        self.step()

    def step(self):
        self.right_arm.set_joint_velocities(self.actions)


    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        angles = dict(zip(self.right_arm.joint_names(),
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.right_arm.move_to_joint_positions(angles)

    def getEndEffectorPosition(self):
        return self.right_arm.endpoint_pose()["position"]

