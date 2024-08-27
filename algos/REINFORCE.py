import torch
import numpy as np
from torch.optim import Adam
from tinygym import GymEnv
from algos import RLAlgorithm
from model import MLPCategorical, MLPGaussian

class REINFORCE(RLAlgorithm):
  def __init__(self, lr=1e-2):
    self.lr = lr
    
  @staticmethod
  def get_discounted_(rewards, gamma=0.99):
    returns = []
    running_return = 0
    for r in reversed(rewards):
      running_return = r + gamma * running_return
      returns = [running_return] + returns
    return torch.tensor(returns)

  def evaluate_cost(self, model, obs, act, rewards):
    logp, _ = model.get_logprob(obs, act)
    rets = self.get_discounted_(rewards) # calculate returns after each rollout
    return -(logp * rets).mean() # pseudoloss

  def train(self, env, hidden_sizes=[32], max_evals=1000): # episodic, bs=1
    model_class = MLPCategorical if env.is_act_discrete else MLPGaussian
    model = model_class(env.n_obs, hidden_sizes, env.n_act)
    optimizer = Adam(model.parameters(), lr=self.lr)
    hist = []
    for i in range(max_evals):
      optimizer.zero_grad()
      ep_obs, ep_acts, ep_rew, *_ = env.rollout(model)
      loss = self.evaluate_cost(model, torch.tensor(np.array(ep_obs)), torch.tensor(np.array(ep_acts)), torch.tensor(np.array(ep_rew)))
      loss.backward()
      optimizer.step()
      if i % 10 == 0:
        avg_reward = sum(ep_rew) / 10
        hist.append((i, avg_reward))
        print(f"nevs {i} reward {avg_reward}")
    return model, hist
