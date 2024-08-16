# test that all algos are working on all tasks running for 10 evals
from tinygym import train, sample, get_available_algos
from pathlib import Path

# test on basic tasks
gym_tasks = ['CartPole-v1', 'LunarLander-v2']

# test custom tasks
def get_available_tasks():
  return [f.stem for f in Path('tasks').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']

for task in get_available_tasks():
  for algo in get_available_algos():
    print(f'running {algo} on {task}')
    try:
      best_model, hist = train(task, algo, max_evals=10)
      rewards, info_dict = sample(task, best_model, n_samples=1, render_mode=None)
    except Exception as e:
      print(f'{algo} failed on {task} with {e}')
    print('----------------\n')
