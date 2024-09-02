# rl-playground

simple environment for training RL algos on custom envs

## Usage

To train and save model:

1. add algo to `algos/`
2. if custom gym env: add to `envs/` and register env in `envs/__init__.py`
3. run: `python tinygym.py --algo [algo] --task [task] --max_evals [default=1000] --save [True]`

To test algos on sample tasks: `python unit_test.py`
ex. `python tinygym.py --algo CMAES --task CartPole-v1 --max_evals 10000`

## Implementations

Includes some basic, minimal implements of RL algos, converges to basic controls tasks in <1000 episodes.

- [x] reinforce (~35 lines of code)
- [x] vpg (~50 lines)
- [x] cma
- [x] ppo (based on SB3)
- [ ] dqn
- [ ] sac
