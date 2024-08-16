# test that all algos are working on all tasks running for 10 evals
from tinygym import train, sample, get_available_algos
from pathlib import Path

def get_available_tasks():
  gym_tasks = ['CartPole-v1', 'LunarLander-v2', 'Pendulum-v1']
  custom_tasks = [f.stem for f in Path('tasks').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']
  return gym_tasks + custom_tasks

def main():
  for task in get_available_tasks():
    for algo in get_available_algos():
      print(f'testing {algo} on {task}')
      best_model, hist = train(task, algo, max_evals=20)
      rewards, info_dict = sample(task, best_model, n_samples=1, render_mode=None)
      print("PASS")
      print('----------------\n')

if __name__ == '__main__':
  main()