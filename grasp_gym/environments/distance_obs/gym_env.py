import gym
import numpy as np
from gymnasium import spaces
from grasp_gym.environments.models.sim_env import SimEnv

class RobotGraspGym(gym.Env):

    '''Position control can change the position, but not the orientation. Also it decides when to grasp by itself'''

    def __init__(self, render_gui=False):
        
        self.sim_env = SimEnv(render_gui)
        self.robot = self.sim_env.robot

        self.max_ts = 1000
        self.episode_ts = 0

        self.success_threshold = 0.025
        self.grasp_success = 0

        self.action_space, self.observation_space = self.define_spaces()

    def define_spaces(self):

        # Define continuous action space for x, y, and z
        action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), 
                                              high=np.array([1.0, 1.0, 1.0, 1.0]), 
                                              dtype=np.float32)

        # Define continuous observation space for distance in x, y, and z
        observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]), 
                                                high=np.array([np.inf, np.inf, np.inf, np.inf]), 
                                                dtype=np.float32)

        return action_space, observation_space


    def reset(self, **kwargs):
        '''Reset the environment and return the initial observation'''
        self.episode_ts = 0
        self.grasp_success = 0
        self.sim_env.reset()
        observations = self.get_observation()
        return observations, {}  # Assuming reset_infos are not needed
        
    def get_observation(self):

        # Calculate distance between robot and ball
        distance = self.sim_env.get_distance()

        gripper_status = self.sim_env.robot.get_gripper_status()
        
        observation = np.concatenate([distance, np.array([gripper_status], dtype=np.float32)], dtype=np.float32)

        return observation

    def step(self, action):
        
        self.sim_env.run_simulation(action)

        observation = self.get_observation()
        reward = self.reward()
        done = self.terminate_episode()

        self.episode_ts += 1
    
        return observation, reward, done, done, {}
    
    
 
    
    
    

        






    
    

    
   

   
    



        


