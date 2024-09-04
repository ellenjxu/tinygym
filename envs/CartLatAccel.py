# https://github.com/ellenjxu/cartlataccel/blob/main/gym_cartlataccel/env.py

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import interp1d
from envs.lib.noise import SimNoise

class CartLatAccelEnv(gym.Env):
  """
  BatchedCartLatAccel env

  Action space: ndarray shape (bs,) representing accel applied to cart
  Observation space: ndarray shape (bs, 3) with cart state and target, [pos, velocity, target_pos]
  Rewards: r = -error/500, where error is abs x-x_target. Scaled to (-10,0)

  Starting state: random state in obs space
  Episode truncation: 500 timesteps
  """

  metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 50,
  }

  def __init__(self, render_mode: str = None, noise_mode: str = None, moving_target: bool = True, env_bs: int = 1):
    self.force_mag = 10.0 # steer -> accel
    self.tau = 0.02  # Time step
    self.max_u = 10.0 # steer/action
    self.max_v = 5.0 # init small v
    self.max_x = 10.0 # max x to clip
    self.max_x_frame = 2.2 # size of render frame

    self.bs = env_bs
    # Action space is continuous steer/accel
    self.action_space = spaces.Box(
      low=-self.max_u, high=self.max_u, shape=(self.bs, 1), dtype=np.float32
    )

    # Obs space is [pos, velocity, target]
    obs_low = np.stack([np.array([-self.max_x, -self.max_v, -self.max_x]) for _ in range(self.bs)])
    self.observation_space = spaces.Box(
      low=obs_low,
      high=-obs_low,
      shape=(self.bs, 3),
      dtype=np.float32
    )

    self.render_mode = render_mode
    self.screen = None
    self.clock = None

    self.max_episode_steps = 500
    self.curr_step = 0
    self.noise_mode = noise_mode
    self.moving_target = moving_target

    np.random.seed(42)

  def generate_traj(self, n_traj=1, n_points=10):
    # generates smooth curve using cubic interpolation
    t_control = np.linspace(0, self.max_episode_steps - 1, n_points)
    control_points = np.random.uniform(-2, 2, (n_traj, n_points)) # slightly less than max x
    f = interp1d(t_control, control_points, kind='cubic')
    t = np.arange(self.max_episode_steps)
    return f(t)

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.state = self.np_random.uniform(
      low=[-self.max_x_frame, -self.max_v, -self.max_x_frame],
      high=[self.max_x_frame, self.max_v, self.max_x_frame],
      size=(self.bs, 3)
    )

    if self.moving_target:
      self.x_targets = self.generate_traj(self.bs)
    else:
      self.x_targets = np.full((self.bs, self.max_episode_steps), self.state[-1]) # fixed target
    self.noise_model = SimNoise(self.max_episode_steps, 1/self.tau, self.noise_mode, seed=seed)

    self.curr_step = 0
    if self.render_mode == "human":
      self.render()
    return np.array(self.state, dtype=np.float32), {}

  def step(self, action):
    x = self.state[:,0]
    v = self.state[:,1]
    action = action.squeeze()
    noisy_action = self.noise_model.add_lat_noise(self.curr_step, action)

    new_a = noisy_action * self.force_mag # steer * force
    new_x = 0.5 * new_a * self.tau**2 + v * self.tau + x
    new_x = np.clip(new_x, -self.max_x, self.max_x)
    new_v = new_a * self.tau + v
    new_x_target = self.x_targets[:, self.curr_step]

    self.state = np.stack([new_x, new_v, new_x_target], axis=1)

    error = abs(new_x - new_x_target)
    reward = -error/self.max_episode_steps # scale reward

    if self.render_mode == "human":
      self.render()

    self.curr_step += 1
    truncated = self.curr_step >= self.max_episode_steps
    info = {"action": action, "noisy_action": noisy_action, "x": new_x, "x_target": new_x_target}
    return np.array(self.state, dtype=np.float32), reward.item(), False, truncated, info

  def render(self):
    if self.screen is None:
      pygame.init()
      if self.render_mode == "human":
        pygame.display.init()
        self.screen = pygame.display.set_mode((600, 400))
      else:  # rgb_array
        self.screen = pygame.Surface((600, 400))
    if self.clock is None:
      self.clock = pygame.time.Clock()

    self.surf = pygame.Surface((600, 400))
    self.surf.fill((255, 255, 255))

    # Only render the first episode in the batch
    first_cart_x = int((self.state[0, 0] / self.max_x_frame) * 300 + 300)  # center is 300
    first_target_x = int((self.x_targets[0, self.curr_step] / self.max_x_frame) * 300 + 300)

    pygame.draw.rect(self.surf, (0, 0, 0), pygame.Rect(first_cart_x - 10, 180, 20, 40))  # cart
    pygame.draw.circle(self.surf, (255, 0, 0), (first_target_x, 200), 5)  # target
    pygame.draw.line(self.surf, (0, 0, 0), (0, 220), (600, 220))  # line

    self.screen.blit(self.surf, (0, 0))
    if self.render_mode == "human":
      pygame.event.pump()
      self.clock.tick(self.metadata["render_fps"])
      pygame.display.flip()
    elif self.render_mode == "rgb_array":
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
      )

  def close(self):
    if self.screen is not None:
      import pygame
      pygame.display.quit()
      pygame.quit()

# if __name__ == "__main__":
#   from stable_baselines3.common.env_checker import check_env
#   env = CartLatAccelEnv()
#   check_env(env)
#   print(env.observation_space, env.action_space)