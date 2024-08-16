import torch
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

def evaluate_cost(model, obs, act, weights):
  logp = model.get_logprob(obs, act)
  return -(logp * weights).mean()

def train_single_epoch(model, env, epoch_steps=1000):
  obs, acts, rews, weights, step_count, n_eps = [], [], [], [], 0, 0 # batch

  while True:
    ep_obs, ep_acts, ep_rews, _ = env.rollout(model)
    obs.extend(ep_obs)
    acts.extend(ep_acts)
    rews.append(sum(ep_rews))
    weights.extend(get_discounted_(ep_rews))
    step_count += len(ep_rews)
    n_eps += 1
    if step_count > epoch_steps:
      break

  epoch_loss = evaluate_cost(model, torch.tensor(obs), torch.tensor(acts), torch.tensor(weights))
  avg_reward = sum(rews) / n_eps
  return epoch_loss, avg_reward, n_eps

def train(task, hidden_sizes=[32], max_evals=1000, epoch_steps=1000, lr=1e-2, seed=None):
  env = GymEnv(task, seed=seed)
  model_class = MLPCategorical if env.is_act_discrete else MLPGaussian
  model = model_class(env.n_obs, hidden_sizes, env.n_act)
  optimizer = Adam(model.parameters(), lr=lr)
  hist, nevs = [], 0

  while True:
    optimizer.zero_grad()
    epoch_loss, avg_reward, num_eps = train_single_epoch(model, env, epoch_steps)
    epoch_loss.backward()
    optimizer.step()

    nevs += num_eps
    hist.append((nevs, avg_reward))
    print(f"nevs {nevs} loss {epoch_loss.item():.3f} reward {avg_reward:.3f}")
    if nevs > max_evals:
      break

  return model, hist
