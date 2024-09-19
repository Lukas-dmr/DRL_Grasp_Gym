import os
import argparse
import gymnasium
import grasp_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from grasp_gym.helper_scripts.custom_callbacks import CustomCallback
from grasp_gym.helper_scripts.helper_funtions import read_hyperparameters_from_yaml
from networks.lstm_policy import CustomActorCriticPolicy, CustomLSTMFeaturesExtractor

def run_env(stage_nr=4, path=os.getcwd() + "/trained_agents/mlp_agent.zip"):
    """
    Run the trained agent in the environment.

    Args:
        stage_nr (int): The stage number of the environment (1-4).
        path (str): The path to the trained agent model.
    """

    # Create and wrap the custom Gym environment
    env = gymnasium.make('GraspEnv-s'+stage_nr, render_gui=True)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(path)

    obs = env.reset()

    while True:
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)

def train_env(stage_nr=4, load_agent=False, agent_name=""):
    """
    Train the agent in the environment.

    Args:
        stage_nr (int): The stage number of the environment (1-4).
        load_agent (bool): Whether to load a previously trained agent.
        agent_name (str): The name of the agent to load.
    """

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
        model = PPO("MlpPolicy", env, verbose=1, **hyperparameters)

    # Define the callback to save checkpoints during training
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir)

     # Define a custom callback to track progress
    custom_callback = CustomCallback(verbose=2)

    # Train the agent with PPO
    model.learn(total_timesteps=int(1e6), callback=[checkpoint_callback, custom_callback], progress_bar=True)

    # Save the final trained model
    model.save("final_model")

    # Close the environment
    env.close()

def test_env(stage_nr=4):
    """
    Test the environment for debugging purposes.

    Args:
        stage_nr (int): The stage number of the environment (1-4).
    """

    env = gymnasium.make('GraspEnv-s'+stage_nr, render_gui=True)
    env.reset()

    tmp = False
    ts = 0
    while True:
        # get cube position
        cube_pos = env.sim_env.get_object_position()

        # get gripper position
        gripper_pos = env.sim_env.robot.get_tcp_position()

        # drive gripper to cube
        action = cube_pos - gripper_pos

        if abs(action[0]) > 0.01 or abs(action[1]) > 0.01:
           observation, reward, done, _, _ = env.step([action[0], action[1], 0, 0])
        elif abs(action[2]) > 0.001:
            observation, reward, done, _, _ = env.step([0, 0, action[2], 0])
        else: 
           observation, reward, done, _, _ =  env.step([0, 0, 0, 1]) 

        if done or ts > 200:
            env.reset()
            ts = 0

        ts += 1

def main():

    parser = argparse.ArgumentParser(description='train, run or test agent')
    # Choose the action to perform (train, run, test)
    parser.add_argument('--action', choices=['train', 'run', 'test'], help='Type of action: train, run, test', default='run')
    # Choose the stage number (1, 2, 3, 4)
    parser.add_argument('--stage', type=str, choices=["1", "2", "3", "4", "t"], default=4, help='Stage number (1, 2, 3, 4)')
    # Choose the checkpoint to load
    parser.add_argument('--checkpoint', default=None, help='Checkpoint name')

    args = parser.parse_args()

    if args.action == 'train':
        if args.checkpoint:
            train_env(stage_nr=str(args.stage), load_agent=True, agent_name=str(args.checkpoint))
        else:
            train_env(stage_nr=str(args.stage))
    elif args.action == 'run':
        run_env(stage_nr=str(args.stage), path=os.path.join(os.getcwd(), 'trained_agents/'+str(args.checkpoint)))
    elif args.action == 'test':
        test_env(stage_nr=str(args.stage))
    else:
        print("Invalid command. Please choose 'train', 'run', or 'test'.")

if __name__ == "__main__":
    main()


