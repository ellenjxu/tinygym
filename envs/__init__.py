from gymnasium.envs.registration import register

register(
  id="CartLatAccel-v0",
  entry_point="envs.CartLatAccel:CartLatAccelEnv"
)