import time
import numpy as np
import torch.nn as nn
from typing import Dict, Any
from dataclasses import field
from pydantic import BaseModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
from tinygym import GymEnv
from model import MLPCategorical, MLPGaussian

class PPOConfig(BaseModel):
  policy: str = 'MlpPolicy'
  n_steps: int = 256 # 2048
  batch_size: int = 256 # 64
  n_epochs: int = 20
  gae_lambda: float = 0.95
  gamma: float = 0.99
  ent_coef: float = 0.01
  learning_rate: float = 1e-3
  clip_range: float = 0.2
  vf_coef: float = 0.5
  max_grad_norm: float = 0.5
  policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {"net_arch": [32]}) # [64, 64]
  normalize_advantage: bool = True
  verbose: int = 0

cfg = PPOConfig()

class HistoryCallback(BaseCallback): # custom callback for hist plot
  def __init__(self, task, verbose=0):
    super(HistoryCallback, self).__init__(verbose)
    self.history = []
    self.n_eps = 0
    self.episode_rewards = []
    self.current_rewards = None
    self.current_lengths = None
    self.task = task

  def _on_training_start(self) -> None:
    self.current_rewards = np.zeros(self.training_env.num_envs)
    self.current_lengths = np.zeros(self.training_env.num_envs)

  def _on_step(self) -> bool:
    rewards = self.locals['rewards']
    dones = self.locals['dones']
    self.current_rewards += rewards
    self.current_lengths += 1

    for i, done in enumerate(dones):
      if done:
        self.n_eps += 1
        self.episode_rewards.append(self.current_rewards[i])

        if self.n_eps % 10 == 0: # save
          avg_reward = np.mean(self.episode_rewards[-10:])
          self.history.append((self.n_eps, avg_reward))
          if self.verbose:
            print(f"eps {self.n_eps}, avg_reward {avg_reward:.2f}, time {time.time()-start:.2f}") #, eval_reward {eval_reward:.2f}")

        self.current_rewards[i] = 0
        self.current_lengths[i] = 0
    return True

def make_env(task, seed=None):
  return lambda: GymEnv(task, seed=seed).env

def load_model(task, ppo_model): # TODO: simplify extract ppo best model -> MLP policy
  env = GymEnv(task)
  model_class = MLPCategorical if env.is_act_discrete else MLPGaussian
  model = model_class(env.n_obs, [32], env.n_act)
  feature_layer = ppo_model.policy.mlp_extractor.policy_net
  action_layer = ppo_model.policy.action_net
  action_net = nn.Sequential(
    *list(feature_layer.children()),
    action_layer
  )
  model.mlp = action_net.cpu()
  return model

def train(task, hidden_sizes, max_evals=1000, n_envs=32, seed=None):
  vec_env = SubprocVecEnv([make_env(task, seed=seed) for _ in range(n_envs)])
  callback = HistoryCallback(task, verbose=1)
  stop_callback = StopTrainingOnMaxEpisodes(max_episodes=max_evals//n_envs, verbose=1)

  ppo_model = PPO(env=vec_env, **cfg.__dict__)
  global start
  start = time.time()
  ppo_model.learn(total_timesteps=np.inf, callback=[callback, stop_callback])
  # ppo_model.save("out/ppo_model")
  best_model = load_model(task, ppo_model)
  return best_model, callback.history
