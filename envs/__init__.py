from gymnasium.envs.registration import register

register(
  id="CartLatAccel-v0",
  entry_point="envs.cartlataccel:CartLatAccelEnv"
)

register(
  id="CartPoleCont-v0",
  entry_point="envs.cartpolecont:CartPoleContEnv"
)