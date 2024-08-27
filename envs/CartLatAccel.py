import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from scipy.interpolate import interp1d
from envs.lib.noise import SimNoise

class CartLatAccelEnv(gym.Env):
    """
    CartTraj

    Action space: ndarray shape (1,) representing accel applied to cart
    Observation space: ndarray shape (3,) with cart state and target, [pos, velocity, target_pos]
    Rewards: r = -error, where error is abs x-x_target. Min is ~0, max is unbounded for each time step (500 rollout). Solved if total reward ~-50

    Starting state: random state in obs space
    Episode truncation: 500 timesteps
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: str = None, noise_mode: str = None, moving_target: bool = True):
        self.force_mag = 5.0 # steer -> accel
        self.tau = 0.02  # Time step
        self.max_u = 5.0 # steer/action
        self.max_x = 2.2
        self.max_v = 1.0 # init small v

        # Action space is continuous steer/accel
        self.action_space = spaces.Box(
            low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32
        )

        # Obs space is [pos, velocity, target]
        self.observation_space = spaces.Box(
            low=np.array([-self.max_x, -self.max_v, -self.max_x]),
            high=np.array([self.max_x, self.max_v, self.max_x]),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.max_episode_steps = 500
        self.curr_step = 0
        self.noise_mode = noise_mode
        self.moving_target = moving_target

    def generate_trajectory(self, n_points=10):
        # generates smooth curve using cubic interpolation
        t_control = np.linspace(0, self.max_episode_steps - 1, n_points)
        control_points = np.random.uniform(-2, 2, n_points) # slightly less than max x
        f = interp1d(t_control, control_points, kind='cubic')
        t = np.arange(self.max_episode_steps)
        traj = f(t)
        return np.clip(traj, -self.max_x, self.max_x)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.np_random.uniform(
            low=[-self.max_x, -self.max_v, -self.max_x],
            high=[self.max_x, self.max_v, self.max_x],
            size=(3,)
        )

        if self.moving_target:
            self.x_targets = self.generate_trajectory()
        else:
            self.x_targets = np.full(self.max_episode_steps, self.state[-1]) # fixed target
        self.noise_model = SimNoise(self.max_episode_steps, 1/self.tau, self.noise_mode, seed=seed)

        self.curr_step = 0
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        x, v, x_target = self.state
        action = action[0]
        noisy_action = self.noise_model.add_lat_noise(self.curr_step, action)

        new_a = noisy_action * self.force_mag # steer * force
        new_x = 0.5 * new_a * self.tau**2 + v * self.tau + x
        new_x = np.clip(new_x, -10., 10.)
        new_v = new_a * self.tau + v
        new_x_target = self.x_targets[self.curr_step]

        self.state = [new_x, new_v, new_x_target]

        error = abs(new_x - new_x_target)
        reward = -error

        if self.render_mode == "human":
            self.render()

        self.curr_step += 1
        truncated = self.curr_step >= self.max_episode_steps
        info = {"action": action, "noisy_action": noisy_action, "x": new_x, "x_target": new_x_target}
        return np.array(self.state, dtype=np.float32), reward, False, truncated, info

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

        # convert to pixel coords
        cart_x = int((self.state[0] / self.max_x) * 300 + 300)  # center is 300
        target_x = int((self.state[-1] / self.max_x) * 300 + 300)

        pygame.draw.rect(self.surf, (0, 0, 0), pygame.Rect(cart_x - 10, 180, 20, 40)) # cart
        pygame.draw.circle(self.surf, (255, 0, 0), (target_x, 200), 5) # target
        pygame.draw.line(self.surf, (0, 0, 0), (0, 220), (600, 220)) # line

        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    @staticmethod
    def plot_traj(info, filename, save_plot=False):
        plt.plot(info[0]["x"], label='actual pos')
        plt.plot(info[0]["x_target"], label='target pos')
        plt.title("actual vs target trajectory")
        plt.ylim([-2.2,2.2])
        plt.legend(loc="upper left")
        if save_plot:
            plt.savefig(f'out/{filename}_traj.png')
        plt.show()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
