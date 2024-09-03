import numpy as np
import pybullet as p
from gymnasium import spaces
from grasp_gym.environments.models.sim_env import SimEnv
from grasp_gym.environments.distance_obs.gym_env import RobotGraspGym

class TestVidGym(RobotGraspGym):

    def __init__(self, render_gui=False):
        
        self.sim_env = SimEnv(render_gui)
        self.robot = self.sim_env.robot

        self.max_ts = 1000
        self.episode_ts = 0

        self.success_threshold = 0.03
        self.grasp_success = 0

        self.action_space, self.observation_space = self.define_spaces()

    def define_spaces(self):

        # Define continuous action space for x, y, and z
        action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                              high=np.array([1.0, 1.0]), 
                                              dtype=np.float32)

        # Define continuous observation space for distance in x, y, and z
        #observation_space = spaces.Box(low=np.array([-np.inf, -np.inf]), 
        #                                        high=np.array([np.inf, np.inf]), 
        #                                        dtype=np.float32)
        

        #observation_space = spaces.Box(low=0.0, high=1.0, shape=(94, 94), dtype=np.float32)

        observation_space = spaces.Dict({
            'depth_image': spaces.Box(low=0, high=255, shape=(94,94), dtype=np.float32),
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        
        return action_space, observation_space
    
    def get_observation(self):

        observation = {}
        observation['depth_image'] = np.array(self.sim_env.get_depth_img(), dtype=np.float32) 
        observation['position'] = np.array(self.sim_env.get_distance()[:2], dtype=np.float32)
        
        return observation
    
    def step(self, action):

        action = np.array([action[0], action[1], 0.0, 0.0])
        
        self.sim_env.run_simulation(action)

        observation = self.get_observation()
        reward = self.reward()
        done = self.terminate_episode()

        self.episode_ts += 1
    
        return observation, reward, done, done, {}

    def reward(self):
        current_distance = self.sim_env.get_distance()
        reward = -np.linalg.norm(current_distance)
        if self.check_reach_success():
            reward = 500
        return reward
    
    def terminate_episode(self):
        if self.episode_ts >= self.max_ts:
            return True
        if self.check_reach_success():
            return True
        return False
    
    def check_reach_success(self):

        distance = self.sim_env.get_distance()

        success = True

        x_dist = abs(distance[0])
        y_dist = abs(distance[1])

        if x_dist > self.success_threshold or y_dist > self.success_threshold:
            success = False

        return success

class StageOneGym(RobotGraspGym):
    
    def reward(self):
        current_distance = self.sim_env.get_distance()
        reward = -np.linalg.norm(current_distance)
        if self.check_reach_success():
            reward = 500
        return reward

    def terminate_episode(self):
        if self.episode_ts >= self.max_ts:
            return True
        if self.check_reach_success():
            return True
        return False
    
    def check_reach_success(self):

        distance = self.sim_env.get_distance()

        success = True

        x_dist = abs(distance[0])
        y_dist = abs(distance[1])
        z_dist = abs(distance[2])

        if x_dist > self.success_threshold or y_dist > self.success_threshold:
            success = False

        if z_dist > 0.06:
            success = False

        return success
    
class StageTwoGym(RobotGraspGym):
    def reward(self):
        
        current_distance = self.sim_env.get_distance()
        
        grasp_penalty = 1
        
        if self.sim_env.robot.get_gripper_status() == 1:
            grasp_penalty = 2
        
        reward = -np.linalg.norm(current_distance)*grasp_penalty
        
        if self.check_reach_success() and self.sim_env.robot.get_gripper_status() == 1:
            reward = 500

        return reward

    def terminate_episode(self):
        if self.episode_ts >= self.max_ts:
            return True
        if self.check_reach_success() and self.sim_env.robot.get_gripper_status() == 1:
            return True
        return False
    
    def check_reach_success(self):

        distance = self.sim_env.get_distance()

        success = True

        x_dist = abs(distance[0])
        y_dist = abs(distance[1])
        z_dist = abs(distance[2])

        if x_dist > self.success_threshold or y_dist > self.success_threshold:
            success = False

        if z_dist > 0.04:
            success = False

        return success
    
class StageThreeGym(RobotGraspGym):
    
    def reward(self):
        current_distance = self.sim_env.get_distance()

        grasp_penalty = 1
        if self.sim_env.robot.get_gripper_status() == 1:
            grasp_penalty = 2

        reward = -np.linalg.norm(current_distance)*grasp_penalty
        self.check_grasp_success()
        
        if self.grasp_success == 1:
            reward = 500
            
        return reward

    def terminate_episode(self):
        if self.episode_ts >= self.max_ts:
             return True
        if self.grasp_success == 1:
             return True
        if not self.sim_env.check_obj_pos():
             return True
        return False
    
    def check_grasp_success(self):
        '''Check if the object is in the gripper'''
        
        # Check position of finger joints
        finger_joint_states = [p.getJointState(self.sim_env.robot.gripper, i)[0] for i in [4, 5]]

        if finger_joint_states[0] < 0.01 or finger_joint_states[1] < 0.01:
            return
        else:
            contactPoints = p.getContactPoints(bodyA=self.sim_env.robot.gripper, bodyB=self.sim_env.obj)
    
            if len(contactPoints) == 0:
                return
            else:
                contact_finger1 = False
                contact_finger2 = False

                for cp in contactPoints:
                    if cp[3] == 4 and cp[4] == -1:
                      contact_finger1 = True
                    if cp[3] == 5 and cp[4] == -1:
                       contact_finger2 = True

                if contact_finger1 and contact_finger2: 
                    self.grasp_success = 1
                    return
                    
                   
class StageFourGym(RobotGraspGym):
    
    def reward(self):
        current_distance = self.sim_env.get_distance()

        grasp_penalty = 1
        if self.sim_env.robot.get_gripper_status() == 1:
            grasp_penalty = 2

        reward = -np.linalg.norm(current_distance)*grasp_penalty
        self.check_grasp_success()
        
        if self.grasp_success == 1:
            reward = 500
            
        return reward

    def terminate_episode(self):
        if self.episode_ts >= self.max_ts:
             return True
        if self.grasp_success == 1:
             return True
        if not self.sim_env.check_obj_pos():
             return True
        return False      

    def check_grasp_success(self):
        '''Check if the object is in the gripper'''
        
        # Check position of finger joints
        finger_joint_states = [p.getJointState(self.sim_env.robot.gripper, i)[0] for i in [4, 5]]

        if finger_joint_states[0] < 0.01 or finger_joint_states[1] < 0.01:
            return
        else:
            contactPoints = p.getContactPoints(bodyA=self.sim_env.robot.gripper, bodyB=self.sim_env.obj)
    
            if len(contactPoints) == 0:
                return
            else:
                contact_finger1 = False
                contact_finger2 = False

                for cp in contactPoints:
                    if cp[3] == 4 and cp[4] == -1:
                      contact_finger1 = True
                    if cp[3] == 5 and cp[4] == -1:
                       contact_finger2 = True

                if contact_finger1 and contact_finger2: 

                    # lift object to validate the grasp
                    self.sim_env.robot.lift_object()
                    
                    obj_pos = self.sim_env.get_object_position()
                
                    if obj_pos[2] > 0.055:
                        self.grasp_success = 1
                        return

