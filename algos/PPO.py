import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyTensorStorage
from algos import RLAlgorithm
from tinygym import GymEnv
from model import ActorCritic

class PPO(RLAlgorithm):
  def __init__(self, lr=3e-4, clip_range=0.2, epochs=10, n_steps=2048, ent_coeff=0.001, bs=64, device='cpu', debug=False, shared_layers=True):
    self.lr = lr
    self.clip_range = clip_range
    self.epochs = epochs
    self.n_steps = n_steps
    self.ent_coeff = ent_coeff
    self.bs = bs
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000, device=device), batch_size=bs)
    self.hist = []
    self.start = time.time()
    self.device = device
    self.debug = debug
    self.eps = 0
    self.shared_layers = shared_layers

  @staticmethod
  def compute_gae(rewards, values, done, next_value, gamma=0.99, lam=0.99):
    returns, advantages = np.zeros_like(rewards), np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + gamma*next_value*(1-done[t]) - values[t]
      gae = delta + gamma*lam*(1-done[t])*gae
      advantages[t] = gae
      returns[t] = gae + values[t]
      next_value = values[t]
    return returns, advantages

  def evaluate_cost(self, model, states, actions, returns, advantages, logprob):
    new_logprob, entropy = model.actor.get_logprob(states, actions)
    ratio = torch.exp(new_logprob-logprob).squeeze()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = nn.MSELoss()(model.critic(states).squeeze(), returns.squeeze())
    entropy_loss = -self.ent_coeff * entropy.mean()
    return {"actor": actor_loss, "critic": critic_loss, "entropy": entropy_loss}

  def train(self, env, hidden_sizes, max_evals=10000):
    model = ActorCritic(env.n_obs, {"pi": hidden_sizes, "vf": hidden_sizes}, env.n_act, env.is_act_discrete, shared_layers=self.shared_layers).to(self.device)
    optimizer = optim.Adam(model.parameters(), lr=self.lr)

    while True:
      # rollout
      start = time.perf_counter()
      states, actions, rewards, dones, _, next_state = env.rollout(model.actor)
      rollout_time = time.perf_counter()-start

      # compute gae
      start = time.perf_counter()
      with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor(np.array(actions)).to(self.device) if env.is_act_discrete else torch.FloatTensor(np.array(actions)).to(self.device)
        values = model.critic(state_tensor).cpu().numpy().squeeze()
        next_values = model.critic(next_state_tensor).cpu().numpy().squeeze()
        logprobs_tensor, _ = model.actor.get_logprob(state_tensor, action_tensor)
        returns, advantages = self.compute_gae(np.array(rewards), values, np.array(dones), next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      gae_time = time.perf_counter()-start

      # add to buffer
      start = time.perf_counter()
      episode_dict = TensorDict(
        {
          "states": state_tensor,
          "actions": action_tensor,
          "returns": torch.FloatTensor(returns).to(self.device),
          "advantages": torch.FloatTensor(advantages).to(self.device),
          "logprobs": logprobs_tensor,
        },
        batch_size=state_tensor.shape[0]
      )
      self.replay_buffer.extend(episode_dict)
      buffer_time = time.perf_counter() - start

      # update
      if len(self.replay_buffer) > self.n_steps:
        start = time.perf_counter()
        for _ in range(self.epochs):
          for i, batch in enumerate(self.replay_buffer):
            costs = self.evaluate_cost(model, batch['states'], batch['actions'], batch['returns'], batch['advantages'], batch['logprobs'])
            loss = costs["actor"] + 0.5 * costs["critic"] + costs["entropy"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == self.n_steps // self.bs:
              break
        self.replay_buffer.empty() # clear buffer
        update_time = time.perf_counter() - start

        if self.debug:
          print(f"critic loss {costs['critic'].item():.3f} entropy {costs['entropy'].item():.3f} mean action {np.mean(abs(np.array(actions)))}")
          print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")
        avg_reward = np.sum(rewards)
        self.hist.append((self.eps, avg_reward))
        print(f"eps {self.eps:.2f}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")

      self.eps += 1 # env bs
      if self.eps > max_evals:
        break
    return model.actor.cpu(), self.hist
