from gymnasium.envs.registration import register
from grasp_gym.environments.distance_obs.gym_stages import StageOneGym, StageTwoGym, StageThreeGym, StageFourGym, TestVidGym



register(id='GraspEnv-s1',
         entry_point= StageOneGym,
)

register(id='GraspEnv-s2',
         entry_point= StageTwoGym,
)

register(id='GraspEnv-s3',
         entry_point= StageThreeGym,
)

register(id='GraspEnv-s4',
         entry_point= StageFourGym,
)

register(id='GraspEnv-st',
         entry_point= TestVidGym,
)