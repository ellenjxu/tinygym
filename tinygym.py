import torch
import pickle
import argparse
import importlib
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from pathlib import Path

class GymEnv:
  def __init__(self, task="CartPole-v1", render_mode=None, seed=None):
    if task in gym.envs.registry.keys():
      self.env = gym.make(task, render_mode=render_mode)
    else:
      module = importlib.import_module(f'tasks.{task}') # custom task
      self.env = module.CustomEnv(render_mode=render_mode)

    self.is_act_discrete = True if isinstance(self.env.action_space, gym.spaces.Discrete) else False
    self.n_obs = self.env.observation_space.shape[-1]
    self.n_act = self.env.action_space.n if self.is_act_discrete else self.env.action_space.shape[-1]
    self.env.reset()
    if seed is not None:
      self.seed(seed)

  def seed(self, seed):
    self.env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

  def rollout(self, model, max_steps=1000, deterministic=False):
    states, actions, rewards  = [], [], []
    infos = {}
    state, _ = self.env.reset()

    for _ in range(max_steps):
      action = model.get_action(state, deterministic=deterministic)
      next_state, reward, terminated, truncated, info = self.env.step(action)
      states.append(state)
      actions.append(action)
      rewards.append(reward)

      for k,v in info.items(): # info dict for entire rollout
        if k not in infos: infos[k] = []
        infos[k].append(v)

      state = next_state
      if terminated or truncated:
        break
    return states, actions, rewards, infos

class VecGymEnv:
  def __init__(self, task="CartPole-v1", render_mode=None, n_envs=4, seed=None):
    if task in gym.envs.registry.keys():
      self.env = gym.make(task, render_mode=render_mode) # TODO gym vec
    else:
      module = importlib.import_module(f'tasks.{task}') # custom task
      self.env = AsyncVectorEnv([lambda: module.CustomEnv() for _ in range(n_envs)])

    self.is_act_discrete = True if isinstance(self.env.action_space, gym.spaces.Discrete) else False
    self.n_obs = self.env.observation_space.shape[-1]
    self.n_act = self.env.action_space.n if self.is_act_discrete else self.env.action_space.shape[-1]
    self.n_envs = n_envs

    self.env.reset()
    if seed is not None:
      self.seed(seed)

  def seed(self, seed):
    self.env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

  def rollout(self, model, max_steps=1000):
    state, _ = self.env.reset() # TODO: seed
    states, actions, rewards, dones = [], [], [], []

    # with self.env:
    for _ in range(max_steps):
      action = model.get_action(state)
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated | truncated

      states.append(state)
      actions.append(action)
      rewards.append(reward)
      dones.append(done)

      state = next_state
      # if done.any():
      #   break
    return states, actions, rewards, dones, next_state

def get_available_algos():
  return [f.stem for f in Path('algos').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']

def get_hidden_sizes(hidden_sizes):
  if hidden_sizes=="None": return []
  return list(map(int, hidden_sizes.split(',')))

def train(task, algo, hidden_sizes=[32], max_evals=1000, save_model=False, seed=None):
  module = importlib.import_module(f'algos.{algo}')
  best_model, hist = module.train(task, hidden_sizes=hidden_sizes, max_evals=max_evals, seed=seed)

  if save_model:
    file_name = f"out/{task}_{algo}_nevs{max_evals}"
    print(f"saving to {file_name}")
    torch.save(best_model, file_name + '.pt')
    with open(file_name + '_hist.pkl', 'wb') as f:
      pickle.dump(hist, f)

  return best_model, hist

def sample(task, best_model, n_samples=10, render_mode=None, seed=None):
  env = GymEnv(task, render_mode=render_mode, seed=seed)
  rewards, info = [], []
  for i in range(n_samples):
    eps_states, eps_actions, eps_rewards, eps_info = env.rollout(best_model, deterministic=True)
    rewards.append(sum(eps_rewards))
    info.append(eps_info)
  return rewards, info # info for plotting

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="CartPole-v1")
  parser.add_argument("--algo", default="cma", choices=get_available_algos())
  parser.add_argument("--max_evals", type=int, default=1000)
  parser.add_argument("--n_samples", type=int, default=1)
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--render_mode", type=str, default="human", choices=["human", None])
  parser.add_argument("--hidden_sizes", type=str, default="32")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  hidden_sizes = get_hidden_sizes(args.hidden_sizes)
  best_model, hist = train(args.task, args.algo, hidden_sizes, args.max_evals, args.save_model, seed=args.seed)
  rewards, _ = sample(args.task, best_model, args.n_samples, args.render_mode)
  print(f"rewards {rewards} avg {np.mean(rewards)}")
