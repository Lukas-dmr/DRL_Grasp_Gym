import gymnasium
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 128):
        # Call the parent constructor with the features dimension
        super(CustomLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.lstm = th.nn.LSTM(input_size=observation_space.shape[0], hidden_size=features_dim, batch_first=True)
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Ensure observations have a batch and sequence dimension
        observations = observations.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, input_size)
        lstm_out, _ = self.lstm(observations)  # Shape: (batch_size, sequence_length=1, hidden_size)
        return lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs,
            features_extractor_class=CustomLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
