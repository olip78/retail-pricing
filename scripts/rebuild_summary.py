"""Rebuild the main forecast figure from the published aggregate table."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
frame = pd.read_csv(ROOT / 'results/forecast_comparison.csv').set_index('Model')
fig, ax = plt.subplots(figsize=(8,4), layout='constrained')
frame.plot.bar(ax=ax)
ax.set(ylabel='logRMSE (lower is better)', xlabel='', title='Historical demand forecasts: identical rows within each period')
ax.tick_params(axis='x', rotation=0)
out = ROOT / 'generated'
out.mkdir(exist_ok=True)
fig.savefig(out / 'forecast_comparison.png', dpi=160)
print(frame.round(4).to_string())
print('Saved generated/forecast_comparison.png')
