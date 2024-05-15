import numpy as np
import pybullet as p
from grasp_gym.environments.distance_obs.gym_env import RobotGraspGym

class StageOneGym(RobotGraspGym):
    def reward(self):
        current_distance = self.sim_env.get_distance()
        reward = -np.linalg.norm(current_distance)
        if self.check_reach_success():
            reward = 500

    def terminate_episode(self):
        if self.episode_ts >= self.max_ts:
            return True
        if self.check_reach_success():
            return True
        return False
    
class StageTwoGym(RobotGraspGym):
    def reward(self):
        current_distance = self.sim_env.get_distance()
        reward = -np.linalg.norm(current_distance)
        if self.check_reach_success() and self.sim_env.robot.get_gripper_status() == 1:
            reward = 500

    def terminate_episode(self):
        if self.episode_ts >= self.max_ts:
            return True
        if self.check_reach_success() and self.sim_env.robot.get_gripper_status() == 1:
            return True
        return False
    
class StageThreeGym(RobotGraspGym):
    def reward(self):
        current_distance = self.sim_env.get_distance()

        grasp_penalty = 1
        if self.sim_env.robot.get_gripper_status() == 1:
            grasp_penalty = 2

        reward = -np.linalg.norm(current_distance)*grasp_penalty
        
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
                    self.robot.lift_object()
                    
                    obj_pos = self.sim_env.get_object_position()
                
                    if obj_pos[2] > 0.055:
                        self.grasp_success = 1
                        return

