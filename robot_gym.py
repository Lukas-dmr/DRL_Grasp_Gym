from email import policy
import os, inspect

from numpy import average

from rhmi_gym.environments.vision_control.VisControlGymEnv import RhmiGymEnv
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import json
import importlib
import ray
from rhmi_gym.helper_scripts.doc import load_config, load_hyperparam, doc_eval
from rhmi_gym.helper_scripts.training_manager import TrainingManager
from ray.tune.registry import register_env

from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()
# Print only tensorflow errors and no warnings
tf.get_logger().setLevel('ERROR')

def train(args):
    """
    Train an RLib agent using one of the following algorithms dqn, ppo or sac

    Args:
        args: containing agent_name, algo (name of algorithm), (optional) load_model (path to model, which should be loaded)
    """

    # TODO: noch cpu und gpu init i.wie einstellen
    ray.shutdown()
    ray.init(num_gpus=1, num_cpus=4, ignore_reinit_error=True) 

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    config = load_config(currentdir)
    hyperparam = load_hyperparam(currentdir, args.algo) 
    config["project_path"] = currentdir
    config["active_algo"] = args.algo
    
    # import the enviornment (defined in config under ENV_TAG)
    env_file = importlib.import_module("rhmi_gym.environments."+str(config[config["ENV_TAG"]]["path"])+"."+
                                        str(config[config["ENV_TAG"]]["file_name"]))
                    
    custom_env = getattr(env_file, str(config[config["ENV_TAG"]]["class_name"]))

    register_env('RhmiEnv-v0', lambda config: custom_env(config))

    for _ in range(3):
        TrainingManager(args, config, hyperparam)

    #TODO: Noch ein überbleibsel
    #x = 19
    # bei old_1 ist 6 als nächstes dran
    #for i in range(12, x):
    #    hyperparam_eps = {}
    #    hyperparam_eps = hyperparam["config"+str(i)]  
    #    TrainingManager(args, config, hyperparam_eps)


def run(args):
    
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/trained_agents/"+str(args.algo).upper()

    # Get path to config and path to checkpoint
    tmp = args.load_model.split("/")
    if tmp[0] == "": 
        tmp = tmp[1]
        agent_path = currentdir+args.load_model
    else: 
        tmp = tmp[0]
        agent_path = currentdir+"/"+args.load_model

    config_path = currentdir+"/"+tmp
    
    config_agent = load_config(config_path)
    config_agent["project_path"] = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    config_agent["active_algo"] = args.algo
    config_agent["env_param"]["visualize"] = config_agent["evaluation"]["visualize"]
    config_agent.update({"env_config": {"eval": True}})

    # Get environment
    env_file = importlib.import_module("rhmi_gym.environments."+str(config_agent[config_agent["ENV_TAG"]]["path"])+"."+
                                        str(config_agent[config_agent["ENV_TAG"]]["file_name"]))

    env = getattr(env_file, str(config_agent[config_agent["ENV_TAG"]]["class_name"]))
    register_env('RhmiEnv-v0', lambda config: env(config))

    ray.shutdown()
    ray.init(num_cpus=4, num_gpus=1)    

    env = RhmiGymEnv(config_agent)

    # Load saved hyperperameters
    with open(config_path+"/params.json") as json_file:
        saved_HP = json.load(json_file)

    algo = importlib.import_module("ray.rllib.agents."+str(config_agent["active_algo"]))
    trainer_module = getattr(algo, str(config_agent["active_algo"]).upper()+"Trainer")

    # Change some Hyperparmeters
    config_algo = algo.DEFAULT_CONFIG.copy()
    config_algo.update(saved_HP)
    config_algo["env"] = None
    config_algo["callbacks"] = None
    config_algo["observation_space"] = env.observation_space
    config_algo["action_space"] = env.action_space
    config_algo["num_workers"] = 1

    if config_agent["evaluation"]["eval_random"]:
        from gym.spaces import Discrete
        agent = Discrete(5)
    else:
        # Init Trainer and restore checkpoint
        agent = trainer_module(config=config_algo)
        agent.restore(agent_path)


    policys = agent.get_policy() 

    """"
    policys.model.action_model.cnns.get(0).base_model.summary()
    policys.model.q_net.cnns.get(0).base_model.summary()
    policys.model.twin_q_net.cnns.get(0).base_model.summary()
    print(policys.model.twin_q_net.cnns)
    print(policys.model.twin_q_net.one_hot)
    print(policys.model.twin_q_net.flatten_dims)
    print(policys.model.twin_q_net.flatten)
    policys.model.twin_q_net.flatten.get(1).base_model.summary()


    print(policys.model.cnns)
    print(policys.model.one_hot)
    print(policys.model.flatten_dims)
    policys.model.cnns.get(0).base_model.summary()
    policys.model.flatten.get(1).base_model.summary()
    policys.model.logits_and_value_model.summary()
    """

    episode_reward = 0
    done = False   
    obs = env.reset()
    nr_of_steps = []

    while env._episode_cnt < config_agent["evaluation"]["eval_episodes"]:
        if config_agent["evaluation"]["eval_random"]: action = agent.sample()
        else: action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            nr_of_steps.append(env._step_cnt)
            env.reset()


    string = "EVAL:\n \
              noise["+str(config_agent["evaluation"]["noise"])+"] \n \
              unknown_objects["+str(config_agent["evaluation"]["unknown_objects"])+"]\n \
              eval_random["+str(config_agent["evaluation"]["eval_random"])+"]\n \
              nr_steps["+str(config_agent["env_param"]["max_steps_per_episode"])+"]\n \
              nr_episodes["+str(config_agent["evaluation"]["eval_episodes"])+"]\n \
              Average nr of timesteps["+str(sum(nr_of_steps)/len(nr_of_steps))+"]\n \
              Accuracy: "+str(env._success_rate/config_agent["evaluation"]["eval_episodes"])

    doc_eval(agent_path.rsplit("/", 2)[0],string)

    return episode_reward


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    train_pars = subparser.add_parser('train', help=str)
    train_pars.add_argument('--agent_name', type=str, required=True)
    train_pars.add_argument('--algo', type=str, default='DQN', required=True)
    train_pars.set_defaults(func=train)

    run_pars = subparser.add_parser('run', help=str)
    run_pars.add_argument('--agent_name', type=str, required=True)
    run_pars.add_argument('--algo', type=str, default='DQN', required=False)
    run_pars.add_argument('--load_model', type=str, required=False, default=None, help="Path to a previous trained agent/model")
    run_pars.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)
