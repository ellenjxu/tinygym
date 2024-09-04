from gymnasium.envs.registration import register

register(
  id="CartPoleCont-v0",
  entry_point="envs.CartPoleCont:CartPoleContEnv"
)