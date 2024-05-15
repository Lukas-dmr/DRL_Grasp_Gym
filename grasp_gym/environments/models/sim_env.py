import os
import time
import numpy as np
import pybullet as p
from grasp_gym.environments.models.robot_gripper import Robot

class SimEnv():

    def __init__(self, render_gui) -> None:

        self.obj = -1
        
        if render_gui: p.connect(p.GUI)
        else: p.connect(p.DIRECT)

        self.model_path = os.getcwd() + "/grasp_gym/environments/models"
        self.load_world()

    def load_world(self):
        '''load the simulation world'''
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF(self.model_path + "/table/table.urdf", [0,0,-0.6], useFixedBase=int(1))

        if self.obj != -1:
            p.removeBody(self.obj)
        self.obj = (self.obj) = p.loadURDF(self.model_path + "/grasping_objects/cube_small.urdf")
        #p.changeDynamics(self.obj, -1, mass=0)


        self.robot = Robot(self.obj)

    def place_object(self):
        min_val = -0.4
        max_val = 0.4
        undesired_min = -0.1
        undesired_max = 0.1

        # Generate random positions within the desired range
        rand_pos = np.random.uniform(min_val, max_val, size=2)

        # Reject positions falling within the forbidden intervals
        while (undesired_min <= rand_pos[0] <= undesired_max) or (undesired_min <= rand_pos[1] <= undesired_max):
            rand_pos = np.random.uniform(min_val, max_val, size=2)

        
        p.resetBasePositionAndOrientation(self.obj, [rand_pos[0], rand_pos[1], 0.05], [0, 0, 0, 1])
        #p.changeDynamics(self.obj, -1, mass=0)

        
    def reset(self):
        self.robot.reset_robot()
        self.place_object()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

    def run_simulation(self, action):
        '''Run the simulation for 5 steps'''
        self.robot.move_robot(action)
        for _ in range(10):
            p.stepSimulation()
            time.sleep(1./240.)

    def get_object_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.obj)
        return np.array(pos)
    
    def get_distance(self):
        '''Get the distance between the tcp and the object'''
        # Get robot position
        robot_pos = self.robot.get_tcp_position()
        # Get ball position
        ball_pos = self.get_object_position()

        # Calculate distance between robot and ball
        distance = robot_pos - ball_pos

        return distance
    
    def check_obj_pos(self):

        obj_pos = self.get_object_position()

        if abs(obj_pos[0]) > 0.45 or abs(obj_pos[1]) > 0.45:
            return False
        
        return True












        

