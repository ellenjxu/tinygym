import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from tinygym import GymEnv
from model import ActorCritic

class PPO:
  def __init__(self, env, model, lr=1e-3, gamma=0.99, lam=0.95, clip_range=0.2, epochs=1, n_steps=1000, ent_coeff=0.01, bs=32, device='cpu', debug=False):
    self.env = env
    self.model = model.to(device)
    self.gamma = gamma
    self.lam = lam
    self.clip_range = clip_range
    self.epochs = epochs
    self.n_steps = n_steps
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    # self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=50)
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000, device=device), batch_size=bs)
    self.bs = bs
    self.hist = []
    self.start = time.time()
    self.device = device
    self.debug = debug
    self.eps = 0

  def compute_gae(self, rewards, values, done, next_value):
    returns, advantages = np.zeros_like(rewards), np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + self.gamma*next_value*(1-done[t]) - values[t]
      gae = delta + self.gamma*self.lam*(1-done[t])*gae
      advantages[t] = gae
      returns[t] = gae + values[t]
      next_value = values[t]
    return returns, advantages

  def evaluate_cost(self, states, actions, returns, advantages, logprob):
    new_logprob, entropy = self.model.actor.get_logprob(states, actions)
    ratio = torch.exp(new_logprob-logprob).squeeze()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = nn.MSELoss()(self.model.critic(states).squeeze(), returns)
    entropy_loss = -self.ent_coeff * entropy.mean()
    return {"actor": actor_loss, "critic": critic_loss, "entropy": entropy_loss}

  def rollout(self, max_steps=1000, deterministic=False): # TODO: gymenv
    states, actions, rewards, dones  = [], [], [], []
    state, _ = self.env.reset()

    for _ in range(max_steps):
      state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
      action = self.model.actor.get_action(state_tensor, deterministic=deterministic).detach().cpu().numpy()
      next_state, reward, terminated, truncated, info = self.env.step(action)
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      done = terminated or truncated
      dones.append(done)

      state = next_state
      if done:
        break
        # state, _ = self.env.reset()
    return states, actions, rewards, dones, next_state

  def train(self, max_evals=1000):
    while True:
      # rollout
      start = time.perf_counter()
      states, actions, rewards, dones, next_state = self.rollout()
      rollout_time = time.perf_counter()-start

      # compute gae
      start = time.perf_counter()
      with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor(np.array(actions)).to(self.device) if self.model.discrete else torch.FloatTensor(np.array(actions)).to(self.device)
        values = self.model.critic(state_tensor).cpu().numpy().squeeze()
        next_values = self.model.critic(next_state_tensor).cpu().numpy().squeeze()
        logprobs_tensor, _ = self.model.actor.get_logprob(state_tensor, action_tensor)

      returns, advantages = self.compute_gae(np.array(rewards), values, np.array(dones), next_values)
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
            advantages = (batch['advantages']-torch.mean(batch['advantages']))/(torch.std(batch['advantages'])+1e-8)
            costs = self.evaluate_cost(batch['states'], batch['actions'], batch['returns'], advantages, batch['logprobs'])
            loss = costs["actor"] + 0.5 * costs["critic"] + costs["entropy"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i == self.n_steps // self.bs:
              break
        self.replay_buffer.empty() # clear buffer
        # self.scheduler.step() # lr decay
        # print(self.scheduler.get_last_lr())
        update_time = time.perf_counter() - start

        # debug info
        if self.debug:
          print(f"critic loss {costs['critic'].item():.3f} entropy {costs['entropy'].item():.3f} mean action {np.mean(abs(np.array(actions)))}")
          print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")

        avg_reward = np.sum(rewards)
        self.hist.append((self.eps, avg_reward))
        print(f"eps {self.eps:.2f}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")

      if self.eps > max_evals:
        print(f"Total time: {time.time() - self.start}")
        break

      self.eps += 1 # cartlataccel env bs
    return self.model.actor.cpu(), self.hist

def train(task, hidden_sizes, max_evals=10000, seed=None):
  print(f"training ppo with hidden_sizes {hidden_sizes} max_evals {max_evals}")
  env = GymEnv(task, seed=seed)
  model = ActorCritic(env.n_obs, {"pi": hidden_sizes, "vf": [32]}, env.n_act, env.is_act_discrete)
  ppo = PPO(env.env, model)
  return ppo.train(max_evals=max_evals)