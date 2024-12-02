import torch
import numpy as np
from torch.optim import Adam
from tinygym import GymEnv
from algos import RLAlgorithm
from model import MLPCategorical, MLPGaussian

class VPG(RLAlgorithm):
  def __init__(self, lr=1e-2, n_steps=1000):
    self.lr = lr
    self.n_steps = n_steps
    
  @staticmethod
  def get_discounted_(rewards, gamma=0.99):
    returns = []
    running_return = 0
    for r in reversed(rewards):
      running_return = r + gamma * running_return
      returns = [running_return] + returns
    return torch.tensor(returns)

  def evaluate_cost(self, model, obs, act, rets):
    logp, _ = model.get_logprob(obs, act)
    return -(logp * rets).mean()

  def train_single_epoch(self, model, env):
    obs, acts, rews, rets, step_count, n_eps = [], [], [], [], 0, 0 # batch

    while True:
      ep_obs, ep_acts, ep_rews, *_ = env.rollout(model)  # over multiple rollouts
      obs.extend(ep_obs)
      acts.extend(ep_acts)
      rews.append(sum(ep_rews))
      rets.extend(self.get_discounted_(ep_rews))
      step_count += len(ep_rews)
      n_eps += 1
      if step_count > self.n_steps:
        break

    epoch_loss = self.evaluate_cost(model, torch.tensor(np.array(obs)), torch.tensor(np.array(acts)), torch.tensor(np.array(rets)))
    avg_reward = sum(rews) / n_eps
    return epoch_loss, avg_reward, n_eps

  def train(self, env, hidden_sizes=[32], max_evals=1000):
    model_class = MLPCategorical if env.is_act_discrete else MLPGaussian
    model = model_class(env.n_obs, hidden_sizes, env.n_act)
    optimizer = Adam(model.parameters(), lr=self.lr)
    hist, nevs = [], 0

    while True:
      optimizer.zero_grad()
      epoch_loss, avg_reward, num_eps = self.train_single_epoch(model, env)
      epoch_loss.backward()
      optimizer.step()

      nevs += num_eps
      hist.append((nevs, avg_reward))
      print(nevs, epoch_loss, avg_reward)
      print(f"nevs {nevs} loss {epoch_loss.item():.3f} reward {avg_reward:.3f}")
      if nevs > max_evals:
        break
    return model, hist
