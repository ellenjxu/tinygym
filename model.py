import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

def mlp(hidden_sizes, activation=nn.Tanh, output_activation=nn.Identity):
  layers = []
  for j in range(len(hidden_sizes)-1):
    act = activation if j < len(hidden_sizes)-2 else output_activation
    layers += [nn.Linear(hidden_sizes[j], hidden_sizes[j+1]), act()]
  return nn.Sequential(*layers)

class MLPCategorical(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh):
    super(MLPCategorical, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

  def forward(self, x):
    x = torch.FloatTensor(x).unsqueeze(0)
    logits = self.mlp(x)
    return logits

  def get_policy(self, obs):
    logits = self.forward(obs)
    return Categorical(logits=logits)

  def get_action(self, obs, deterministic=False):
    action = self.get_policy(obs).sample()
    if deterministic: # get most likely
      logits = self.forward(obs)
      probs = F.softmax(logits, dim=-1)
      action = torch.argmax(probs, dim=-1)
    return action.detach().cpu().tolist()[0]

  def get_logprob(self, obs, act):
    logprob = self.get_policy(obs).log_prob(act)
    return logprob

class MLPGaussian(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh, log_std=0.):
    super(MLPGaussian, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))

  def forward(self, x):
    x = torch.FloatTensor(x).unsqueeze(0)
    mean = self.mlp(x)
    std = torch.exp(self.log_std)
    covariance_mat = torch.diag(std ** 2)
    return mean, covariance_mat

  def get_policy(self, obs):
    mean, covariance_mat = self.forward(obs)
    return MultivariateNormal(mean, covariance_mat)

  def get_action(self, obs, deterministic=False):
    action = self.get_policy(obs).sample()[0]
    if deterministic: # get mean action
      action = self.forward(obs)[0][0]
    # if len(action) == 1: # TODO: clean
    #   return [action.item()]
    # else:
    return action.detach().cpu().numpy() #.tolist()

  def get_logprob(self, obs, act):
    logprob = self.get_policy(obs).log_prob(act)
    return logprob

class MLPCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh):
    super(MLPCritic, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

  def forward(self, x):
    return self.mlp(x)

class ActorCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh, discrete=False):
    super(ActorCritic, self).__init__()
    self.discrete = discrete
    if discrete:
      self.actor = MLPCategorical(obs_dim, hidden_sizes["pi"], act_dim, activation)
    else:
      self.actor = MLPGaussian(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = MLPCritic(obs_dim, hidden_sizes["vf"])

  def forward(self, x):
    if self.discrete:
      actor_out = self.actor(x)
    else:
      actor_out = self.actor.forward(x)[0]  # mean
    critic_out = self.critic(x)
    return actor_out, critic_out
