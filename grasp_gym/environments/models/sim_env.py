import os
import cv2
import time
import numpy as np
import pybullet as p
import pybullet_data
from grasp_gym.environments.models.robot_gripper import Robot

class SimEnv():
    """
    Class for building the pybullet simulation environment
    """

    def __init__(self, render_gui, fix_object=True) -> None:

        self.obj = -1
        # Set the visualization mode
        self.render = render_gui
        # Set the object to be fixed or not
        self.fix_object = fix_object
        
        if render_gui: p.connect(p.GUI)
        else: p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.model_path = os.getcwd() + "/grasp_gym/environments/models"
        self.load_world()

    def load_world(self):
        """
        Load simulation world with table, object and robot
        """
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF(self.model_path + "/table/table.urdf", [0,0,-0.6], useFixedBase=int(1))

        self.obj = (self.obj) = p.loadURDF(self.model_path + "/grasping_objects/cube_small.urdf")
        
        if self.fix_object:
            p.changeDynamics(self.obj, -1, mass=0)

        # Load the robot
        self.robot = Robot(self.obj, render=self.render)

    def place_object(self):
        """
        Place the object in a random position on the table
        """

        rand_pos_x = np.random.uniform(-0.1, 0.3, size=1)
        rand_pos_y = np.random.uniform(-0.3, 0.3, size=1)
        
        p.resetBasePositionAndOrientation(self.obj, [rand_pos_x, rand_pos_y, 0.05], [0, 0, 0, 1])
        
        if self.fix_object:
            p.changeDynamics(self.obj, -1, mass=0)
   
    def reset(self):
        """
        Reset the simulation environment
        """
        self.robot.reset_robot()
        self.place_object()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

    def run_simulation(self, action):
        """
        Run the simulation with the given action
        """

        self.robot.move_robot(action)
        for _ in range(10):
            p.stepSimulation()
            time.sleep(1./240.)

    def get_object_position(self):
        """
        Returns:
            pos (np.array): The position of the object in world coordinates
        """
        pos, _ = p.getBasePositionAndOrientation(self.obj)
        return np.array(pos)
    
    def get_distance(self):
        """
        Returns:
            distance (np.array): The distance between the robot and the object
        """
        # Get robot position
        robot_pos = self.robot.get_tcp_position()
        # Get cube position
        ball_pos = self.get_object_position()
        # Calculate distance between robot and ball
        distance = robot_pos - ball_pos

        return distance
    
    def check_obj_pos(self):
        """
        Returns:
            bool: Whether the object is within the table boundaries or not
        """

        obj_pos = self.get_object_position()
        if abs(obj_pos[0]) > 0.45 or abs(obj_pos[1]) > 0.45:
            return False
        return True












        

