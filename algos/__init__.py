from torch import nn
from tinygym import GymEnv
from typing import List, Tuple

class RLAlgorithm:
  def train(env: GymEnv, hidden_sizes: List[int], max_evals: int, seed: int) -> Tuple[nn.Module, List[Tuple[int, float]]]:
    """
    Args:
      task: GymEnv wrapper around env
      hidden_sizes: list of hidden dims for policy model mlp
      max_evals: max number of episodes/rollouts during training
    Returns:
      best_model: final trained policy model (e.g. actor)
      hist: list of tuples (eps, rew) for training cost plot
    """
    raise NotImplementedError