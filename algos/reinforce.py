import torch
import numpy as np
from torch.optim import Adam
from tinygym import GymEnv
from model import MLPCategorical, MLPGaussian

def get_discounted_(rewards, gamma=0.99):
  returns = []
  running_return = 0
  for r in reversed(rewards):
    running_return = r + gamma * running_return
    returns = [running_return] + returns
  return torch.tensor(returns)

def evaluate_cost(model, obs, act, rewards):
  logp = model.get_logprob(obs, act)
  rets = get_discounted_(rewards) # calculate returns after each rollout
  return -(logp * rets).mean() # pseudoloss

def train(task, hidden_sizes=[32], max_evals=1000, seed=None):  # episodic, bs=1
  env = GymEnv(task, seed=seed)
  model_class = MLPCategorical if env.is_act_discrete else MLPGaussian
  model = model_class(env.n_obs, hidden_sizes, env.n_act)
  optimizer = Adam(model.parameters(), 1e-2)
  hist = []
  for i in range(max_evals):
    optimizer.zero_grad()
    ep_obs, ep_acts, ep_rew, _ = env.rollout(model)
    loss = evaluate_cost(model, torch.tensor(np.array(ep_obs)), torch.tensor(np.array(ep_acts)), torch.tensor(np.array(ep_rew)))
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
      avg_reward = sum(ep_rew) / 10
      hist.append((i, avg_reward))
      print(f"nevs {i} reward {avg_reward}")
  return model, hist
