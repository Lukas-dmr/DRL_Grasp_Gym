import os
import time
import numpy as np
import pybullet as p
from grasp_gym.environments.models.camera import GripperCamera

GRIPPER_OPEN = 1
GRIPPER_CLOSE = 0

class Robot():
  """
  Class to load and control a 2-finger-gripper
  """

  def __init__(self, cube_id, render=False):
    """
    Args:
        cube_id (int): The ID of the object in the simulation
        render (bool): Whether to render the GUI or not
    """
    self.model_path = os.getcwd() + "/grasp_gym/environments/models"
    self.gripper = self.load_robot()
    self.gripper_cam = GripperCamera(self.gripper, cube_id, render=render)
    self.obj_id = cube_id
    self.camera_angle = 0.0
    self.gripper_status = 0

  def load_robot(self):
    """
    Load the gripper model into the simulation environment
    """
    return p.loadURDF(self.model_path + "/panda_gripper/gripper_model.urdf", [-0.4, 0, 0.4], [ 0, 0, 0, 1 ])

  
  def reset_robot(self):
    """
    Reset the gripper to its initial position
    """

    p.setJointMotorControl2(self.gripper, 0, p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(self.gripper, 1, p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(self.gripper, 2, p.POSITION_CONTROL, targetPosition=0)
    
    p.setJointMotorControl2(self.gripper, 3, p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(self.gripper, 4, p.POSITION_CONTROL, targetPosition=self.camera_angle)
    
    p.setJointMotorControl2(self.gripper, 6, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)
    p.setJointMotorControl2(self.gripper, 7, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)

    self.gripper_status = 0

  def move_robot(self, action):
    """
    Move the gripper to a new target position
    """
    # Scale the action to control the gripper's movement
    scaled_action = np.array(action[:3]) * 0.1
    valid_action = self.keep_boundaries(scaled_action)
    
    # Get the current gripper position
    current_position = self.get_robot_position()

    # Calculate the new target position
    target_position = current_position + valid_action

    # Move the gripper to the target position by setting joint positions
    p.setJointMotorControl2(self.gripper, 0, p.POSITION_CONTROL, targetPosition=target_position[0])
    p.setJointMotorControl2(self.gripper, 1, p.POSITION_CONTROL, targetPosition=target_position[1])
    p.setJointMotorControl2(self.gripper, 2, p.POSITION_CONTROL, targetPosition=target_position[2])

    self.rotate_gripper()
    
    if action[3] > 0 and self.gripper_status == 0: 
       self.close_gripper()
    
    elif action[3] <= 0 and self.gripper_status == 1:
       self.open_gripper()

  def open_gripper(self):
    """
    Open the gripper
    """
    p.setJointMotorControl2(self.gripper, 6, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)
    p.setJointMotorControl2(self.gripper, 7, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)
    self.gripper_status = 0

  def close_gripper(self):
    """
    Close the gripper
    """
    p.setJointMotorControl2(self.gripper, 6, p.POSITION_CONTROL, targetPosition=GRIPPER_CLOSE)
    p.setJointMotorControl2(self.gripper, 7, p.POSITION_CONTROL, targetPosition=GRIPPER_CLOSE)
    self.gripper_status = 1

  def keep_boundaries(self, action):
    """
    Check if the gripper is moving within the valid range, if not adjust action

    Args:
        action (np.array): The action to be checked

    Returns:
        action (np.array): The adjusted action
    """
    current_position = self.get_tcp_position()
    target_position = current_position + action
    
    # Check if the target position is within the valid range
    valid_range = [[-0.5, 0.5], [-0.5, 0.5], [0.02, 0.5]]

    for i in range(3):
        if target_position[i] <= valid_range[i][0]:
            if action[i] < 0:
                action[i] = 0
                
        if target_position[i] >= valid_range[i][1]:
            if action[i] > 0:
                action[i] = 0
               

    return action
  
  def get_tcp_position(self):
    """
    Get the position of the TCP
    """
    tcp_pos, _ = p.getLinkState(self.gripper, 8)[:2]
    return np.array(tcp_pos)

  def get_robot_position(self):
    """
    Get the positions of joints x, y, and z
    """
    joint_positions = []
    for joint_index in [0, 1, 2]:
        joint_state = p.getJointState(self.gripper, joint_index)
        joint_positions.append(joint_state[0])  # Append the position component
    return joint_positions
  
  def get_tcp_ori(self):
    """
    Get the orientation of the TCP
    """
    _, tcp_ori = p.getLinkState(self.gripper, 7)[:2]
    return np.array(tcp_ori)
  
  def rotate_gripper(self):
    """
    Change orientation if distance is small
    """
    cam_pos, _ = p.getLinkState(self.gripper, 3)[:2]
    obj_pos, _ = p.getBasePositionAndOrientation(self.obj_id)
    dist = np.linalg.norm(np.array(cam_pos) - np.array(obj_pos))

    if abs(dist) < 0.5:
      p.setJointMotorControl2(self.gripper, 4, p.POSITION_CONTROL, targetPosition=0)
    else:
      p.setJointMotorControl2(self.gripper, 4, p.POSITION_CONTROL, targetPosition=self.camera_angle)

  def get_gripper_status(self):
    """
    Returns:
        gripper_status (int): The status of the gripper (0: closed, 1: open)
    """
    return self.gripper_status
  
  def lift_object(self):
    """
    Lift the object to validate the grasp
    """
    p.setJointMotorControl2(self.gripper, 2, p.POSITION_CONTROL, targetPosition=0.15, maxVelocity=0.1)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1./240.)

        
    
