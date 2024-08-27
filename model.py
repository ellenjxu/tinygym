import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

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

  def forward(self, x: torch.Tensor):
    assert x.ndim == 2 # need batch dim
    return self.mlp(x)

  def get_policy(self, obs: torch.Tensor):
    logits = self.forward(obs)
    probs = F.softmax(logits, dim=-1)
    return probs

  def get_action(self, obs: torch.Tensor, deterministic=False):
    probs = self.get_policy(obs)
    if deterministic: # get most likely
      action = torch.argmax(probs, dim=-1)
    else:
      action = torch.multinomial(probs, num_samples=1)
    return action.detach().cpu().numpy().squeeze()

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
    logits = self.forward(obs)
    log_probs = F.log_softmax(logits, dim=-1)
    logprob = log_probs.gather(1, act.unsqueeze(-1)).squeeze(-1)
    return logprob

class MLPGaussian(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh, log_std=3.):
    super(MLPGaussian, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))
    # self.register_buffer('std', self.log_std.exp())

  def forward(self, x: torch.Tensor):
    return self.mlp(x)

  def get_action(self, obs: torch.Tensor, deterministic=False):
    mean = self.forward(obs)
    std = self.log_std.exp()
    action = mean[0] if deterministic else torch.normal(mean, std)[0]
    return action.detach().cpu().numpy()

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
    mean = self.forward(obs)
    std = self.log_std.exp()
    logprob = -0.5 * (((act - mean)**2) / std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
    return logprob.sum(dim=-1)

class MLPCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh):
    super(MLPCritic, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

  def forward(self, x: torch.Tensor):
    return self.mlp(x)

class ActorCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, discrete=False):
    super(ActorCritic, self).__init__()
    self.discrete = discrete
    self.actor = MLPGaussian(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = MLPCritic(obs_dim, hidden_sizes["vf"])

  def forward(self, x: torch.Tensor):
    actor_out, _ = self.actor(x) # mean
    critic_out = self.critic(x)
    return actor_out, critic_out