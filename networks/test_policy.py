import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomFeaturesExtractor(nn.Module):
    def __init__(self, observation_space):
        super(CustomFeaturesExtractor, self).__init__()
        
        # Extract dimensions from the observation space
        distance_dim = observation_space['distance'].shape[0]
        gripper_dim = observation_space['gripper_status'].n
        
        # Define neural networks for processing distance and gripper status
        self.distance_network = nn.Sequential(
            nn.Linear(distance_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.gripper_network = nn.Sequential(
            nn.Linear(gripper_dim, 16),
            nn.ReLU()
        )
        
        # Combine features from distance and gripper status
        combined_dim = 32 + 16  # Output dimensions of both networks
        self.combined_network = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
    def forward(self, observation):
        distance_features = self.distance_network(observation['distance'])
        gripper_features = self.gripper_network(observation['gripper_status'].float())  # Convert to float
        combined_features = torch.cat([distance_features, gripper_features], dim=1)
        return self.combined_network(combined_features)
    

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule=None, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule=lr_schedule, **kwargs)
        self.features_extractor = CustomFeaturesExtractor(observation_space)

        if lr_schedule is None:
            lr_schedule = lambda _: 0.001  # Default to a constant learning rate of 0.001 if not provided

        # Create the optimizer using the provided lr_schedule
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)









