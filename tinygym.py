import os
import wandb
import torch
import pickle
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

class GymEnv:
  def __init__(self, task, render_mode=None, seed=None):
    if task in gym.envs.registry.keys():
      self.env = gym.make(task, render_mode=render_mode)
    else:
      module = importlib.import_module(f'tasks.{task}')
      self.env = module.CustomEnv(render_mode=render_mode)
    if render_mode == "rgb_array": # save video
      self.env = RecordVideo(self.env, video_folder="out/", episode_trigger=lambda e: True)

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

  def rollout(self, model, max_steps=1000, deterministic=False, seed=None):
    states, actions, rewards  = [], [], []
    state, _ = self.env.reset(seed=seed)
    infos = {}

    with self.env:
      for _ in range(max_steps):
        action = model.get_action(state, deterministic=deterministic)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        for k,v in info.items():
          if k not in infos: infos[k] = []
          infos[k].append(v)

        state = next_state
        if terminated or truncated:
          break
      return states, actions, rewards, infos

def get_available_algos():
  return [f.stem for f in Path('algos').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']

def get_hidden_sizes(hidden_sizes):
  if hidden_sizes=="None": return []
  return list(map(int, hidden_sizes.split(',')))

def train(task, algo, hidden_sizes=[32], max_evals=1000, save_model=False, seed=None):
  module = importlib.import_module(f'algos.{algo}')
  best_model, hist = module.train(task, hidden_sizes=hidden_sizes, max_evals=max_evals, seed=seed)

  if save_model:
    os.makedirs('out', exist_ok=True)
    filename = f"out/{task}_{algo}_nevs{max_evals}"
    print(f"saving to {filename}")
    torch.save(best_model, filename + '.pt')
    with open(filename + '_hist.pkl', 'wb') as f:
      pickle.dump(hist, f)
    # wandb
    wandb.init(project="tinygym_training", name=f"{task}_{algo}")
    wandb.config.update({ "task": task, "algo": algo, "hidden_sizes": hidden_sizes, "max_evals": max_evals, "seed": seed })
    for step, reward in hist:
      wandb.log({"reward": reward}, step=step)
    wandb.save(filename + '.pt')
  return best_model, hist

def sample(task, model, n_samples=10, render_mode=None, seed=None):
  env = GymEnv(task, render_mode=render_mode, seed=seed)
  rewards, infos = [], []
  for i in range(n_samples):
    eps_states, eps_actions, eps_rewards, eps_info = env.rollout(best_model, deterministic=True, seed=seed)
    rewards.append(sum(eps_rewards))
    infos.append(eps_info)
  return rewards, infos

def plot_traj(info, filename, save_plot=False):
  plt.plot(info[0]["x"], label='actual pos')
  plt.plot(info[0]["x_target"], label='target pos')
  plt.title("actual vs target trajectory")
  plt.ylim([-2.2,2.2])
  plt.legend(loc="upper left")
  if save_plot:
    plt.savefig(f'out/{filename}_traj.png')
    wandb.log({"traj plot": plt})
  plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="CartPole-v1")
  parser.add_argument("--algo", default="cma", choices=get_available_algos())
  parser.add_argument("--max_evals", type=int, default=1000)
  parser.add_argument("--n_samples", type=int, default=1)
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array", "None"])
  parser.add_argument("--hidden_sizes", type=str, default="32")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  hidden_sizes = get_hidden_sizes(args.hidden_sizes)
  best_model, hist = train(args.task, args.algo, hidden_sizes, args.max_evals, args.save_model, seed=args.seed)
  if args.save_model:
    rewards, info = sample(args.task, best_model, args.n_samples, render_mode="rgb_array")
    wandb.log({"video": wandb.Video(f'out/rl-video-episode-0.mp4', fps=50, format="mp4")})
  else:
    rewards, info = sample(args.task, best_model, args.n_samples, args.render_mode)
  print(f"sample run rewards {rewards} avg {np.mean(rewards)}")

  if args.task == "CartLatAccel":
    filename = f"{args.task}_{args.algo}_nevs{args.max_evals}"
    plot_traj(info, filename, save_plot=args.save_model)