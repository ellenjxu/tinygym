import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch.optim import RMSprop
from torchrl.data import ReplayBuffer, LazyTensorStorage
from algos import RLAlgorithm
from model import MLPQNetwork

class DQN(RLAlgorithm):
  def __init__(self, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995, bs=64, action_bins=11, device='cpu', debug=False):
    self.lr = lr
    self.gamma = gamma
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.bs = bs
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=50000, device=device), batch_size=bs)
    self.action_bins = action_bins
    self.device = device
    self.debug = debug

  def get_action_values(self, env):
    if env.is_act_discrete:
      return None, env.n_act
    if env.is_act_bounded and env.n_act == 1:
      action_values = np.linspace(env.act_bound[0], env.act_bound[1], self.action_bins)
      return action_values, len(action_values)
    raise ValueError("DQN only supports discrete actions or 1D bounded Box actions")

  def update(self, model):
    batch = self.replay_buffer.sample()
    states = batch['state'].to(self.device)
    actions = batch['action'].to(self.device)
    rewards = batch['reward'].to(self.device)
    next_states = batch['next_state'].to(self.device)
    dones = batch['done'].to(self.device)

    # target = r + gamma * max_a' Q(s', a'; theta)
    q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
      next_q_values = model(next_states).max(dim=-1).values
      targets = rewards + self.gamma * (1 - dones) * next_q_values

    # TODO: clip TD error? Smooth L1 Loss?
    loss = nn.MSELoss()(q_values, targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()

  def train(self, env, hidden_sizes=[32], max_evals=1000):
    action_values, n_actions = self.get_action_values(env)
    model = MLPQNetwork(env.n_obs, hidden_sizes, n_actions, action_values=action_values).to(self.device)
    self.optimizer = RMSprop(model.parameters(), lr=self.lr)  # TODO: Adam vs RMSProp (used in original DQN paper)

    hist = []
    eps = self.eps_start

    for i in range(max_evals):
      state, _ = env.env.reset()
      ep_reward = 0

      for _ in range(1000):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_idx = model.get_action_idx(state_tensor, eps=eps)
        action = model.action_idx_to_env_action(action_idx)
        next_state, reward, terminated, truncated, _ = env.env.step(action)
        done = terminated or truncated
        transition = TensorDict(
          {
            'state': torch.FloatTensor(state).unsqueeze(0),
            'action': torch.LongTensor([int(action_idx)]),
            'reward': torch.FloatTensor([reward]),
            'next_state': torch.FloatTensor(next_state).unsqueeze(0),
            'done': torch.FloatTensor([done]),
          },
          batch_size=[1]
        )
        self.replay_buffer.extend(transition)

        eps = max(self.eps_end, eps * self.eps_decay)

        if len(self.replay_buffer) >= self.bs:
          loss = self.update(model)

        ep_reward += reward
        state = next_state
        if done:
          break

      hist.append((i + 1, ep_reward))
      print(f"eps {i+1:.2f}, reward {ep_reward:.3f}, epsilon {eps:.3f}")

    return model.cpu(), hist
