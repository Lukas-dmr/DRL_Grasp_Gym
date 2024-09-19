import numpy as np
import pybullet as p
from grasp_gym.environments.gym_env import RobotGraspGym

"""
This file contains the different stages of the gym environment, for curriculum learning of the robot grasping task.
"""

class StageOneGym(RobotGraspGym):
    """
    Stage 1 of the robot grasping task

    The robot has to reach only to minimize the distance until it reaches a certain threshold to get the reward.
    """
    
    def reward(self):
        """
        Returns:
            reward (float): The reward of the environment
        """
        current_distance = self.sim_env.get_distance()
        reward = -np.linalg.norm(current_distance)
        if self.check_reach_success():
            reward = 500
        return reward

    def terminate_episode(self):
        """
        Returns:
            done (bool): Whether the episode is done or not
        """
        if self.episode_ts >= self.max_ts:
            return True
        if self.check_reach_success():
            return True
        return False
    
    def check_reach_success(self):
        """
        Returns:
            success (bool): Whether the robot has reached the object or not
        """

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
    """
    Stage 2 of the robot grasping task

    The robot has to reach the distance threshold and close the gripper inside of the threshold to get the reward.
    """
    def reward(self):
        """
        Returns:
            reward (float): The reward of the environment
        """
        current_distance = self.sim_env.get_distance()
        
        grasp_penalty = 1
        # Check if the gripper is closed
        if self.sim_env.robot.get_gripper_status() == 1:
            # Penalty to avoid closing the gripper before reaching the object
            grasp_penalty = 2
        
        reward = -np.linalg.norm(current_distance)*grasp_penalty
        
        if self.check_reach_success() and self.sim_env.robot.get_gripper_status() == 1:
            reward = 500

        return reward

    def terminate_episode(self):
        """
        Returns:
            done (bool): Whether the episode is done or not
        """
        if self.episode_ts >= self.max_ts:
            return True
        if self.check_reach_success() and self.sim_env.robot.get_gripper_status() == 1:
            return True
        return False
    
    def check_reach_success(self):
        """
        Returns:
            success (bool): Whether the robot has reached the object and the gripper is closed or not
        """

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
    """
    Stage 3 of the robot grasping task

    The robot has to reach the distance threshold, close the gripper inside of the threshold and touch the object with its fingers to get the reward.
    """
    
    def reward(self):
        """
        Returns:
            reward (float): The reward of the environment
        """
        current_distance = self.sim_env.get_distance()

        grasp_penalty = 1
        # Check if the gripper is closed
        if self.sim_env.robot.get_gripper_status() == 1:
            # Penalty to avoid closing the gripper before reaching the object
            grasp_penalty = 2

        reward = -np.linalg.norm(current_distance)*grasp_penalty
        self.check_grasp_success()
        
        if self.grasp_success == 1:
            reward = 500
            
        return reward

    def terminate_episode(self):
        """
        Returns:
            done (bool): Whether the episode is done or not
        """
        if self.episode_ts >= self.max_ts:
             return True
        if self.grasp_success == 1:
             return True
        if not self.sim_env.check_obj_pos():
             return True
        return False
    
    def check_grasp_success(self):
        '''
        Returns:
            grasp_success (bool): Whether the robot is touching the object with its fingers or not
        '''
        
        # Check position of finger joints
        finger_joint_states = [p.getJointState(self.sim_env.robot.gripper, i)[0] for i in [6, 7]]
        
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
                    if cp[3] == 6 and cp[4] == -1:
                      contact_finger1 = True
                    if cp[3] == 7 and cp[4] == -1:
                       contact_finger2 = True

                if contact_finger1 and contact_finger2: 
                    self.grasp_success = 1
                    return
                                      
class StageFourGym(RobotGraspGym):
    """
    Stage 4 of the robot grasping task

    The robot has to grasp the object and lift it to a certain height to get the reward.
    """
    
    def reward(self):
        """
        Returns:
            reward (float): The reward of the environment
        """
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
        """
        Returns:
            done (bool): Whether the episode is done or not
        """
        if self.episode_ts >= self.max_ts:
             return True
        if self.grasp_success == 1:
             return True
        if not self.sim_env.check_obj_pos():
             return True
        return False      

    def check_grasp_success(self):
        '''
        Returns:
            grasp_success (bool): Whether the robot is lifting the object or not
        '''
        
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





