import os
import time
import numpy as np
import pybullet as p

GRIPPER_OPEN = 1
GRIPPER_CLOSE = 0

class Robot():
  """
  Class to load and control a 3-finger-gripper without a roboter
  """

  def __init__(self, cubeId):
    self.model_path = os.getcwd() + "/grasp_gym/environments/models"
    self.gripper = self.load_robot()
    self.cube = cubeId

    self.grasp_success = 0
    self.gripper_status = 0

  def load_robot(self):
    """
    Load the gripper model into the simulation environment
    """
    return p.loadURDF(self.model_path + "/panda_gripper/gripper_model.urdf", [0, 0, 0.2], [ 0, 0, 0, 1 ])

  
  def reset_robot(self):
    """
    Reset the gripper to its initial position
    """

    p.setJointMotorControl2(self.gripper, 0, p.POSITION_CONTROL, targetPosition=0)
    p.setJointMotorControl2(self.gripper, 1, p.POSITION_CONTROL, targetPosition=0)
    
    p.setJointMotorControl2(self.gripper, 2, p.POSITION_CONTROL, targetPosition=0.2)
    p.setJointMotorControl2(self.gripper, 3, p.POSITION_CONTROL, targetPosition=0)
    
    #p.resetBasePositionAndOrientation(self.gripper, [0, 0, 0.2], [ 0, 0, 0, 1 ])
    p.setJointMotorControl2(self.gripper, 4, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)
    p.setJointMotorControl2(self.gripper, 5, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)
    self.grasp_success = 0
    self.gripper_status = 0

  def move_robot(self, action):
    """
    Move the gripper to a new position
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
    
    if action[3] > 0 and self.gripper_status == 0: 
       self.close_gripper()
    
    elif action[3] <= 0 and self.gripper_status == 1: 
       self.open_gripper()

  def open_gripper(self):
    """
    Open the gripper
    """
    p.setJointMotorControl2(self.gripper, 4, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)
    p.setJointMotorControl2(self.gripper, 5, p.POSITION_CONTROL, targetPosition=GRIPPER_OPEN)
    self.gripper_status = 0

  def close_gripper(self):
    """
    Close the gripper
    """
    p.setJointMotorControl2(self.gripper, 4, p.POSITION_CONTROL, targetPosition=GRIPPER_CLOSE)
    p.setJointMotorControl2(self.gripper, 5, p.POSITION_CONTROL, targetPosition=GRIPPER_CLOSE)
    self.gripper_status = 1

  def keep_boundaries(self, action):
    """
    Check if the gripper is moving within the valid range, if not adjust action
    """
    current_position = self.get_tcp_position()
    target_position = current_position + action

    """ print("current ", current_position)
    print("action ", action[:3])
    print("target ", target_position) """
    
    # Check if the target position is within the valid range
    valid_range = [[-0.5, 0.5], [-0.5, 0.5], [0.02, 0.25]]

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
    tcp_pos, _ = p.getLinkState(self.gripper, 6)[:2]
    return np.array(tcp_pos)

  def get_robot_position(self):
    """
    Get the positions of joints 1, 2, and 3
    """
    joint_positions = []
    for joint_index in [0, 1, 2]:
        joint_state = p.getJointState(self.gripper, joint_index)
        joint_positions.append(joint_state[0])  # Append the position component
    return joint_positions

  def get_gripper_status(self):
    """
    Get the status of the gripper
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

        
    
