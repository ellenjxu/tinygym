import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from pydantic import BaseModel
from gymnasium.wrappers import RecordVideo
from tinygym import GymEnv, train, sample

class ReportConfig(BaseModel):
  task: str = "CartPole-v1"
  algo: str = "PPO"
  max_evals: int = 1000
  hidden_sizes: list = [32]
  save_model: bool = False
  n_runs: int = 5
  n_samples: int = 1
  seed: int = 42
  out: str = f"out/{task}_{algo}_report"

def plot_loss_curve(history, out_path="out/loss_curve.html"):
  '''creates html plotly curve'''
  steps, rewards = zip(*history)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=steps, y=rewards, mode='lines', name='Reward'))
  fig.update_layout(
    xaxis_title="nevs",
    yaxis_title="reward",
    showlegend=True
  )
  fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
  with open(out, 'w') as f:
    f.write(fig_html)

def encode_base64(gif_path="out/rl-video-episode-0.mp4"):
  with open(gif_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')
  return encoded

def make_report(cfg: ReportConfig):
  print(f"Training {cfg.algo} on {cfg.task}")
  best_model, history = train(cfg.task, cfg.algo, cfg.hidden_sizes, cfg.max_evals, save=cfg.save_model, seed=cfg.seed)
  rewards, info = sample(cfg.task, best_model, cfg.n_samples, "rgb_array")

  # generate report
  report_path = "out/report.html"
  loss_plot_path = "loss_curve.html"
  vid_base64 = encode_base64()

  with open(report_path, 'w') as f:
    f.write(f"<html><body>\n")
    f.write(f"<h1>Report for {cfg.task} using {cfg.algo}</h1>\n")
    f.write(f"<pre>{cfg.dict()}</pre>\n")

    f.write(f"<h2>Final reward after {cfg.max_evals} evaluations: {history[-1][1]}</h2>\n")
    f.write(f"<h2>Loss Curve</h2>\n")
    f.write(f'<iframe src="{loss_plot_path}" width="100%" height="600"></iframe>\n')

    f.write(f"<h2>Rollout Video</h2>\n")
    f.write(f'<video width="640" height="480" controls>\n')
    f.write(f'<source src="data:video/mp4;base64,{vid_base64}" type="video/mp4">\n')
    f.write(f"Your browser does not support the video tag.\n")
    f.write(f'</video>\n')
    f.write(f"</body></html>\n")

  print(f"report saved at {report_path}.")

if __name__ == "__main__":
  cfg = ReportConfig()
  make_report(cfg)