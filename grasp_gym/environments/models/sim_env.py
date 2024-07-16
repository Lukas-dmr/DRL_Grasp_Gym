import os
import cv2
import time
import numpy as np
import pybullet as p
from grasp_gym.environments.models.robot_gripper import Robot

import pybullet_data



class SimEnv():

    def __init__(self, render_gui, fix_object=False) -> None:

        self.obj = -1
        self.fix_object = fix_object
        self.render = render_gui
        
        if render_gui: p.connect(p.GUI)
        else: p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.visual_marker_id = -1

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
        
        if self.fix_object:
            p.changeDynamics(self.obj, -1, mass=0)


        self.robot = Robot(self.obj, render=self.render)

    def place_object(self):
        min_val = -0.1
        max_val = 0.4
        undesired_min = -0.1
        undesired_max = 0.1

        # Generate random positions within the desired range
        rand_pos = np.random.uniform(min_val, max_val, size=2)

        #TODO Removed
        # Reject positions falling within the forbidden intervals
        #while (undesired_min <= rand_pos[0] <= undesired_max) or (undesired_min <= rand_pos[1] <= undesired_max):
        #    rand_pos = np.random.uniform(min_val, max_val, size=2)

        
        p.resetBasePositionAndOrientation(self.obj, [0.4, 0, 0.05], [0, 0, 0, 1])
        
        if self.fix_object:
            p.changeDynamics(self.obj, -1, mass=0)
   
    def reset(self):
        self.robot.reset_robot()
        self.place_object()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

    def run_simulation(self, action):
        '''Run the simulation for 5 steps'''

        tcp_pos, _ = p.getLinkState(self.robot.gripper, 4)[:2]
        # Get ball position
        ball_pos = self.get_object_position()

        # Calculate distance between robot and ball
        dist = tcp_pos - ball_pos
        rot = 1
        print(dist)
        if abs(dist[0]) < 0.2 and abs(dist[1]) < 0.2:
            rot = 0

        self.robot.move_robot(action, rotation=rot)
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
    
    def get_depth_img(self):
        return self.robot.gripper_cam.get_depth_image()

    def check_obj_pos(self):

        obj_pos = self.get_object_position()

        if abs(obj_pos[0]) > 0.45 or abs(obj_pos[1]) > 0.45:
            return False
        
        return True












        

