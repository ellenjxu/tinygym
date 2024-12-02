from pathlib import Path
from tinygym import train, sample, get_available_algos

def main():
  for task in ['CartPole-v1', 'LunarLander-v2', 'Pendulum-v1', 'CartLatAccel-v1']:
    for algo in get_available_algos():
      print(f'testing {algo} on {task}')
      best_model, hist = train(task, algo, max_evals=20)
      rewards, info_dict = sample(task, best_model, n_samples=1, render_mode=None)
      print("PASS")
      print('----------------\n')

if __name__ == '__main__':
  main()