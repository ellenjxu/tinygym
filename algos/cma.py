import torch
import numpy as np
from torch import nn
from cmaes import CMA
from functools import partial
from multiprocessing import Pool
from tinygym import GymEnv
from model import MLPCategorical, MLPGaussian

def update_params(model, p):
  i = 0
  layers = [module for module in model if isinstance(module, nn.Linear)]
  for layer in layers:
    n = layer.weight.numel()
    new_weights = torch.tensor(p[i:i+n], dtype=torch.float32).view(layer.out_features, layer.in_features)
    layer.weight = nn.Parameter(new_weights)
    i += n
    if layer.bias is not None:
      n = layer.bias.numel()
      new_biases = torch.tensor(p[i:i+n], dtype=torch.float32)
      layer.bias = nn.Parameter(new_biases)
      i += n

def evaluate_fitness(model, task, sample_size, params, seed=None):
  env = GymEnv(task, seed=seed)
  update_params(model.mlp, params)
  costs = []
  for i in range(sample_size):
    _, _, rewards, *_ = env.rollout(model)
  cost = -sum(rewards)
  costs.append(cost)
  return np.mean(costs)

def train(task, hidden_sizes, max_evals=1000, seed=None, popsize=25, sigma=1e-1, sample_size=10, max_workers=32):
  env = GymEnv(task, seed=seed)
  model_class = MLPCategorical if env.is_act_discrete else MLPGaussian
  model = model_class(env.n_obs, hidden_sizes, env.n_act)
  n_params = sum(p.numel() for p in model.parameters())
  optimizer = CMA(mean=np.zeros(n_params), sigma=sigma, population_size=popsize)
  hist = []
  nevs = 0

  with Pool(processes=max_workers) as pool:
    while True:
      solutions = []
      params = [optimizer.ask() for _ in range(optimizer.population_size)]
      values = pool.map(partial(evaluate_fitness, model, task, sample_size, seed=seed), params)
      solutions = list(zip(params, values))
      optimizer.tell(solutions)
      mindx = np.argmin(values)

      nevs += popsize # TODO: testing; *sample_size
      print(f'nevs {nevs} reward {-np.mean(values)} best {-values[mindx]}')
      hist.append((nevs, -np.mean(values))) # (nev, rewards)

      if nevs >= max_evals:
        break

  update_params(model.mlp, params[mindx]) # best model
  return model, hist
