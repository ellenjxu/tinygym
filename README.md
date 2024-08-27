# rl-playground

simple environment for training RL algos on custom envs

## Usage

To train, simply add algos to `algos/` and tasks to `envs/`, then run:
`python tinygym.py --algo [algo] --task [task] --max_evals [default=1000] --save [True]`

To test algos on sample tasks: `python unit_test.py`

## Implementations

Includes some basic, minimal implements of RL algos.

- [x] reinforce (~35 lines of code)
- [x] vpg (~50 lines)
- [x] cma
- [x] ppo (based on SB3)
- [ ] dqn
- [ ] sac
