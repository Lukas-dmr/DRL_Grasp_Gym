from gymnasium.envs.registration import register

register(id='GraspEnv-v0',
         entry_point='grasp_gym.environments:RobotGraspGym',
)
