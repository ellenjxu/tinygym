# tinygym

A tiny RL playground with minimal, hackable implementations of common RL algorithms.

1. `algos/`: custom implementations of common RL algorthms
2. `envs/`: custom gym environments

```
python tinygym.py --algo [algo] --task [task] --max_evals [default=1000] --save [True]
```

Test on sample tasks: `python unit_test.py`

## RL

- [x] reinforce (~35 lines of code)
- [x] vpg (~50 lines)
- [x] cma
- [x] ppo (based on SB3)
- [ ] dqn
- [ ] sac

Converges to basic controls tasks in <1K episodes (CMA takes longer, ~10K).
