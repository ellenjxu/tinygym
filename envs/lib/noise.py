'''Noise model for CartLatAccel task'''

import numpy as np

def get_no_noise_params():
  return {'lag': 0,             # horizontal (time) shift
          't_corr': [0.5, 5],   # freq (time window size)
          'dy': 0.0,            # vertical shift
          'lat_action': 0.0}    # amplitude (scaled)
CAMERA_FPS = 50                 # task render fps

class SimNoise:
  def __init__(self, n, fps=None, noise_mode=None, seed=0):
    self.n = n
    assert fps is not None
    self.fps = fps
    np.random.seed(seed)

    params = get_no_noise_params()
    if noise_mode == "REALISTIC":
      params['lag'] = 0.5
      params['t_corr'] = [0.5, 5]
      params['dy'] = 1
      params['lat_action'] = 0.3
    elif noise_mode == "HIGH":
      params['lag'] = 1 # 50 timesteps*.02 fps=1s lag
      params['t_corr'] = [0.5, 100] # 5000 ts/rollout*.02=100s, entire rollour correlated
      params['dy'] = 100
      params['lat_action'] = 3

    self.params = params
    # LAG
    self.lag = np.random.uniform(0, params['lag'])
    # NOISE TIME CORRELATION
    self.t_noise_corr = np.random.uniform(*params['t_corr'])
    # LAT
    self.dy_noise = np.random.uniform(-params['dy'], params['dy'], size=self.n)
    self.dy_noise = self.correlate_noise(self.dy_noise)
    self.lat_action_noise = np.random.normal(0.0, params['lat_action'], size=self.n)
    self.lat_action_noise = self.correlate_noise(self.lat_action_noise)

    self.reset()

  def reset(self, init_accel=0.):
    # scale the lag to be in [0, lag*FPS]
    lat_lag_frames = int(self.lag / self.params['lag'] * (int(self.fps*self.params['lag']) + 1)) if self.params['lag'] > 0 else 0
    self.lats = [0 for _ in range(lat_lag_frames)]

  def correlate_noise(self, noise_samples):
    t_noise_corr_frames = int(self.t_noise_corr * CAMERA_FPS) # use CAMERA_FPS here because correlation is done on all frames
    t_noise_corr_kernel = np.ones(t_noise_corr_frames)/np.sqrt(t_noise_corr_frames)
    return np.convolve(noise_samples, t_noise_corr_kernel, mode='same')

  def add_lat_noise(self, step, action):
    dty = (1 + self.lat_action_noise[step]) * action # scaled action
    dty_noise = self.dy_noise[step] / self.fps
    self.lats.append(dty)
    return self.lats.pop(0) + dty_noise