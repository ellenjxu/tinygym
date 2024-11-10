import torch
import numpy as np
from torch import nn
from cmaes import CMA
from functools import partial
from multiprocessing import Pool
from tinygym import GymEnv
from algos import RLAlgorithm
from model import MLPCategorical, MLPGaussian, MLPBeta

class CMAES(RLAlgorithm):
  def __init__(self, popsize=25, sigma=1e-1, sample_size=1):
    self.popsize = popsize
    self.sigma = sigma
    self.sample_size = sample_size

  def update_params(self, model, p):
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

  def evaluate_fitness(self, model, task, params, seed=None):
    env = GymEnv(task, seed=seed)
    self.update_params(model.mlp, params)
    costs = []
    for _ in range(self.sample_size):
      _, _, rewards, *_ = env.rollout(model)
      cost = -sum(rewards)
      costs.append(cost)
    return np.mean(costs)

  def train(self, env, hidden_sizes, max_evals=1000, max_workers=32, seed=None):
    model_class = MLPCategorical if env.is_act_discrete else MLPGaussian
    model = model_class(env.n_obs, hidden_sizes, env.n_act)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = CMA(mean=np.zeros(n_params), sigma=self.sigma, population_size=self.popsize)
    hist = []
    nevs = 0

    with Pool(processes=max_workers) as pool:
      while True:
        params = [optimizer.ask() for _ in range(optimizer.population_size)]
        values = pool.map(partial(self.evaluate_fitness, model, env.task, seed=seed), params)
        solutions = list(zip(params, values))
        optimizer.tell(solutions)
        mindx = np.argmin(values)

        nevs += self.popsize*self.sample_size
        print(f'nevs {nevs} reward {-np.mean(values)} best {-values[mindx]}')
        hist.append((nevs, -np.mean(values)))  # (nev, rewards)

        if nevs >= max_evals:
          break

    self.update_params(model.mlp, params[mindx])  # best model
    return model, hist
