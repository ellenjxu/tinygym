from tinygym import train, sample, get_available_algos

seed = 42

GYM_TASKS = ['CartPole-v1', 'LunarLander-v2', 'Pendulum-v1', 'CartLatAccel-v1']

def test_all_basic():
  """Make sure all algos run on all tasks"""
  for task in GYM_TASKS:
    for algo in get_available_algos():
      print(f'testing {algo} on {task}')
      best_model, hist = train(task, algo, max_evals=20, seed=seed)
      rewards, info_dict = sample(task, best_model, n_samples=1, render_mode=None)
      print("PASS")
      print('----------------\n')

def test_all_cartpole():
  """All tasks should get 500.0 (max reward) on CartPole-v1 (REINFORCE and VPG can be unstable)"""
  # for algo in get_available_algos():
  for algo in get_available_algos():
    if algo == "VPG" or algo == "REINFORCE":
      continue
    print(f'testing {algo} on CartPole-v1')
    max_evals = 1000 if algo != 'CMAES' else 10000
    best_model, hist = train('CartPole-v1', algo, max_evals=max_evals, seed=seed)
    rewards, info_dict = sample('CartPole-v1', best_model, n_samples=5, render_mode=None)
    print(f'rewards: {rewards}')
    assert all(r >= 475.0 for r in rewards), f"Failed to get 475.0 reward on CartPole-v1 with {algo}"
    print("PASS")
    print('----------------\n')

if __name__ == '__main__':
  test_all_basic()
  test_all_cartpole()
  print("ALL TESTS PASSED")
