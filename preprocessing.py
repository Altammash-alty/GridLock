import pandas as pd
import numpy as np
import optuna
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import pygeohash as pgh

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Plot style ─────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0a0',
    'ytick.color':      '#a0a0a0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3a',
    'axes.titlecolor':  '#ffffff',
    'figure.titlesize': 16,
})
PALETTE = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff8cc8']
MISSING_TOKEN = "__MISSING__"


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — EDA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════
def run_eda(train: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("  SECTION 1 — EDA & FEATURE VISUALIZATIONS")
    print("="*60)

    df = train.copy()
    df['hour']      = df['timestamp'].apply(lambda t: int(t.split(':')[0]))
    df['minute']    = df['timestamp'].apply(lambda t: int(t.split(':')[1]))
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15

    # ── Plot 1: Demand Distribution ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Traffic Demand Distribution', fontsize=16, fontweight='bold')

    axes[0].hist(df['demand'], bins=60, color=PALETTE[0],
                 edgecolor='none', alpha=0.85)
    axes[0].set_title('Demand Histogram')
    axes[0].set_xlabel('Demand')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['demand'].mean(), color=PALETTE[1],
                    linestyle='--', linewidth=1.5, label=f"Mean={df['demand'].mean():.3f}")
    axes[0].axvline(df['demand'].median(), color=PALETTE[2],
                    linestyle='--', linewidth=1.5, label=f"Median={df['demand'].median():.3f}")
    axes[0].legend()

    sns.boxplot(x='Weather', y='demand', data=df,
                palette=PALETTE, ax=axes[1])
    axes[1].set_title('Demand by Weather')
    axes[1].set_xlabel('Weather')
    axes[1].set_ylabel('Demand')

    plt.tight_layout()
