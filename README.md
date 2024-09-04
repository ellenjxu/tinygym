# rl-playground

a tiny RL playground for training algos on custom tasks 🤖

## Usage

1. add algo to `algos/`
2. if custom gym env: add to `envs/` and register env in `envs/__init__.py`
3. train and save model:
   `python tinygym.py --algo [algo] --task [task] --max_evals [default=1000] --save [True]`

To check algos on sample tasks: `python unit_test.py`

## Implementations

Includes minimal implements of RL algos.

- [x] reinforce (~35 lines of code)
- [x] vpg (~50 lines)
- [x] cma
- [x] ppo (based on SB3)
- [ ] dqn
- [ ] sac

converges to basic controls tasks in <1K episodes (CMA takes longer, ~10K).
