import os
import sys
import yaml
import argparse
import gymnasium
import grasp_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from networks.test_policy import CustomActorCriticPolicy, CustomLSTMFeaturesExtractor

def read_hyperparameters_from_yaml(file_path=os.getcwd() + "/hyperparameters/ppo.yaml"):
    with open(file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

class CustomCallback(BaseCallback):
    def __init__(self, verbose=2):
        super().__init__(verbose)
        self.cumulative_rewards_single_episode = []  # List to store cumulative rewards for each episode
        self.cumulative_rewards_all_episodes = []  # List to store cumulative rewards for all episodes

    def _on_step(self):

        self.cumulative_rewards_single_episode.append(self.locals['rewards'][0])  # Append the rewards for the current episode

        # Check if an episode has completed
        if self.locals['dones'].any():
            # Calculate the cumulative reward for the current episode
            cumulative_reward = np.mean(self.cumulative_rewards_single_episode)
            self.cumulative_rewards_all_episodes.append(cumulative_reward)
            self.cumulative_rewards_single_episode = []  # Reset the list for the next episode

        # printer after 5000 steps
        if self.num_timesteps % 5000 == 0:
            print(f"Step: {self.num_timesteps}, Mean Reward: {np.mean(self.cumulative_rewards_all_episodes)}")
            self.cumulative_rewards_all_episodes = []

        return True

        


def test_env(stage_nr=4):

    env = gymnasium.make('GraspEnv-s'+stage_nr, render_gui=True)
    env.reset()

    tmp = False
    ts = 0
    
    while True:
        action = env.action_space.sample()
        action = [0.5,0,0,0]
        dist = env.sim_env.get_distance()
        if abs(dist[0]) < 0.05:
            action = [0,0,-0.5,0]
        observation, reward, done, _, _ = env.step(action)
        input()

        """ # get cube position
        cube_pos = env.sim_env.get_object_position()

        # get gripper position
        gripper_pos = env.sim_env.robot.get_tcp_position()

        # drive gripper to cube
        action = cube_pos - gripper_pos

        if not tmp:
            env.step([0, 0, 0, 1])
            tmp = True
        else: 

        if abs(action[0]) > 0.01 or abs(action[1]) > 0.01:
           observation, reward, done, _, _ = env.step([action[0], action[1], 0, 0])
        elif abs(action[2]) > 0.001:
            observation, reward, done, _, _ = env.step([0, 0, action[2], 0])
        else: 
           observation, reward, done, _, _ =  env.step([0, 0, 0, 1]) """

        if done or ts > 200:
            env.reset()
            ts = 0

        ts += 1

def run(stage_nr=4, path=os.getcwd() + "/checkpoints/rl_model_60000_steps.zip"):

    # Create and wrap the custom Gym environment
    env = gymnasium.make('GraspEnv-s'+stage_nr, render_gui=True)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(path)

    obs = env.reset()
    done = False

    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        
def train(stage_nr=4, load_agent=False, agent_name=""):

    # Create the directory if it does not exist
    checkpoint_dir = os.getcwd() + "/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create and wrap the custom Gym environment
    env = gymnasium.make('GraspEnv-s'+stage_nr, render_gui=False)
    env = DummyVecEnv([lambda: env])

    hyperparameters = read_hyperparameters_from_yaml()

    if load_agent:
        model = PPO.load(checkpoint_dir+"/"+agent_name, env=env, **hyperparameters)
    else:
        # Define and configure the PPO agent, CustomActorCriticPolicy, MlpPolicy
        model = PPO(CustomActorCriticPolicy, env, verbose=1, **hyperparameters)

    # Define the callback to save checkpoints during training
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir)

     # Define a custom callback to track progress
    custom_callback = CustomCallback(verbose=2)

    # Train the agent with PPO
    model.learn(total_timesteps=int(1e6), callback=[checkpoint_callback, custom_callback], progress_bar=True)

    # Save the final trained model
    model.save("final_model")

    # Optionally, load the final trained model later
    # loaded_model = PPO.load("final_model")

    # Close the environment
    env.close()

def main():

    parser = argparse.ArgumentParser(description='train, run or test agent')
    parser.add_argument('--action', choices=['train', 'run', 'test'], help='Type of action: train, run, test', default='run')
    parser.add_argument('--stage', type=str, choices=["1", "2", "3", "4"], default=4, help='Stage number (1, 2, 3, 4)')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint name')

    args = parser.parse_args()

    if args.action == 'train':
        if args.checkpoint:
            train(stage_nr=str(args.stage), load_agent=True, agent_name=str(args.checkpoint))
        else:
            train(stage_nr=str(args.stage))
    elif args.action == 'run':
        run(stage_nr=str(args.stage), path=os.path.join(os.getcwd(), 'checkpoints/'+str(args.checkpoint)))
    elif args.action == 'test':
        test_env(stage_nr=str(args.stage))
    else:
        print("Invalid command. Please choose 'train', 'run', or 'test'.")

if __name__ == "__main__":
    main()


