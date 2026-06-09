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
    plt.savefig('plot1_demand_distribution.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot1_demand_distribution.png")

    # ── Plot 2: Demand by Time ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Demand Patterns Over Time', fontsize=16, fontweight='bold')

    hourly = df.groupby('hour')['demand'].mean().reset_index()
    axes[0].plot(hourly['hour'], hourly['demand'],
                 color=PALETTE[0], linewidth=2.5, marker='o', markersize=4)
    axes[0].fill_between(hourly['hour'], hourly['demand'],
                         alpha=0.2, color=PALETTE[0])
    axes[0].set_title('Average Demand by Hour')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Avg Demand')
    axes[0].axvspan(6, 9,   alpha=0.15, color=PALETTE[3], label='Morning peak')
    axes[0].axvspan(17, 21, alpha=0.15, color=PALETTE[1], label='Evening peak')
    axes[0].legend()

    slot_demand = df.groupby('time_slot')['demand'].mean().reset_index()
    axes[1].plot(slot_demand['time_slot'], slot_demand['demand'],
                 color=PALETTE[2], linewidth=1.5, alpha=0.9)
    axes[1].set_title('Average Demand by 15-min Slot')
    axes[1].set_xlabel('Time Slot (0=00:00, 95=23:45)')
    axes[1].set_ylabel('Avg Demand')

    plt.tight_layout()
    plt.savefig('plot2_demand_time.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot2_demand_time.png")

    # ── Plot 3: Demand by Road & Location Features ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Demand by Road & Location Features', fontsize=16, fontweight='bold')

    sns.boxplot(x='RoadType', y='demand', data=df,
                palette=PALETTE, ax=axes[0])
    axes[0].set_title('Demand by Road Type')

    sns.boxplot(x='LargeVehicles', y='demand', data=df,
                palette=[PALETTE[0], PALETTE[1]], ax=axes[1])
    axes[1].set_title('Demand by Large Vehicles')

    sns.boxplot(x='Landmarks', y='demand', data=df,
                palette=[PALETTE[2], PALETTE[3]], ax=axes[2])
    axes[2].set_title('Demand by Landmarks')

    plt.tight_layout()
    plt.savefig('plot3_demand_road_features.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot3_demand_road_features.png")

    # ── Plot 4: Correlation Heatmap ────────────────────────────────
    num_cols = ['demand', 'NumberofLanes', 'Temperature', 'hour', 'time_slot']
    road_map    = {'Highway': 0, 'Street': 1, 'Residential': 2}
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['road_enc']    = df['RoadType'].map(road_map).fillna(-1)
    df['weather_enc'] = df['Weather'].map(weather_map).fillna(-1)
    df['lv_bin']      = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['lm_bin']      = (df['Landmarks'] == 'Yes').astype(int)

    corr_cols = ['demand', 'NumberofLanes', 'Temperature',
                 'hour', 'time_slot', 'road_enc',
                 'weather_enc', 'lv_bin', 'lm_bin']
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    plt.tight_layout()
    plt.savefig('plot4_correlation_heatmap.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot4_correlation_heatmap.png")

    # ── Plot 5: Top geohash demand ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Geospatial Demand Patterns', fontsize=16, fontweight='bold')

    top_geo = df.groupby('geohash')['demand'].mean().nlargest(20)
    axes[0].barh(range(len(top_geo)), top_geo.values,
                 color=PALETTE[0], alpha=0.85)
    axes[0].set_yticks(range(len(top_geo)))
    axes[0].set_yticklabels(top_geo.index, fontsize=8)
    axes[0].set_title('Top 20 Geohashes by Avg Demand')
    axes[0].set_xlabel('Avg Demand')

    demand_by_lanes = df.groupby('NumberofLanes')['demand'].mean().reset_index()
    axes[1].bar(demand_by_lanes['NumberofLanes'].astype(str),
                demand_by_lanes['demand'],
                color=PALETTE[2], alpha=0.85)
    axes[1].set_title('Avg Demand by Number of Lanes')
    axes[1].set_xlabel('Number of Lanes')
    axes[1].set_ylabel('Avg Demand')

    plt.tight_layout()
    plt.savefig('plot5_geo_demand.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot5_geo_demand.png")

    # ── Plot 6: Temperature vs Demand ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Temperature & Weather vs Demand', fontsize=16, fontweight='bold')

    sample = df.dropna(subset=['Temperature']).sample(
        min(3000, len(df)), random_state=42)
    axes[0].scatter(sample['Temperature'], sample['demand'],
                    c=PALETTE[0], alpha=0.3, s=8)
    axes[0].set_title('Temperature vs Demand')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Demand')

    weather_hour = df.groupby(['Weather', 'hour'])['demand'].mean().reset_index()
    for i, weather in enumerate(df['Weather'].dropna().unique()):
        sub = weather_hour[weather_hour['Weather'] == weather]
        axes[1].plot(sub['hour'], sub['demand'],
                     label=weather, color=PALETTE[i % len(PALETTE)],
                     linewidth=2, marker='o', markersize=3)
    axes[1].set_title('Hourly Demand by Weather')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Avg Demand')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('plot6_temp_weather_demand.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot6_temp_weather_demand.png")

    print("\n  All plots saved in current directory.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "NumberOfLanes" in result and "NumberofLanes" not in result:
        result = result.rename(columns={"NumberOfLanes": "NumberofLanes"})

    # --- Timestamp ---
    parts            = result["timestamp"].astype(str).str.split(":", n=1, expand=True)
    result["hour"]   = pd.to_numeric(parts[0], errors="coerce")
    result["minute"] = pd.to_numeric(parts[1], errors="coerce")

    # --- Day features ---
    numeric_day           = pd.to_numeric(result["day"], errors="coerce")
    result["day_of_week"] = numeric_day % 7
    result["is_weekend"]  = result["day_of_week"].isin([5, 6]).astype(float)

    # --- Time block ---
    result["time_slot"] = (
        result["hour"] * 4 + result["minute"] // 15
    ).astype("Int64")

    # --- Cyclical encodings ---
    result["minutes_since_midnight"] = result["hour"] * 60 + result["minute"]
    radians              = 2 * np.pi * result["minutes_since_midnight"] / (24 * 60)
    result["hour_sin"]   = np.sin(radians)
    result["hour_cos"]   = np.cos(radians)
    minute_radians       = 2 * np.pi * result["minute"] / 60
    result["minute_sin"] = np.sin(minute_radians)
    result["minute_cos"] = np.cos(minute_radians)
    day_radians          = 2 * np.pi * result["day_of_week"] / 7
    result["day_sin"]    = np.sin(day_radians)
    result["day_cos"]    = np.cos(day_radians)

    # --- Road / weather ---
    large_vehicle_allowed = result["LargeVehicles"].eq("Allowed").astype(float)
    numeric_lanes         = pd.to_numeric(result["NumberofLanes"], errors="coerce")
    numeric_temperature   = pd.to_numeric(result["Temperature"], errors="coerce")

    weather_severity = (
        result["Weather"]
        .fillna(MISSING_TOKEN)
        .map({"Sunny": 0.0, "Rainy": 1.0, "Foggy": 1.5, "Snowy": 2.0})
        .fillna(0.5)
    )

    result["capacity_proxy"]           = numeric_lanes * large_vehicle_allowed
    result["temp_weather_interaction"]  = numeric_temperature * weather_severity
    result["lanes_weather"]             = numeric_lanes * weather_severity

    result["road_weather_interaction"] = (
        result["RoadType"].fillna(MISSING_TOKEN).astype(str)
        + "__"
        + result["Weather"].fillna(MISSING_TOKEN).astype(str)
    )
    result["geohash_hour_interaction"] = (
        result["geohash"].fillna(MISSING_TOKEN).astype(str)
        + "__"
        + result["hour"].fillna(-1).astype(int).astype(str)
    )
    result["geohash_weather"] = (
        result["geohash"].fillna(MISSING_TOKEN).astype(str)
        + "__"
        + result["Weather"].fillna(MISSING_TOKEN).astype(str)
    )
    result["road_enc"]    = result["RoadType"].map(
        {"Highway": 0, "Street": 1, "Residential": 2}).fillna(-1).astype(int)
    result["weather_enc"] = result["Weather"].map(
        {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}).fillna(-1).astype(int)
    result["lv_bin"]      = large_vehicle_allowed.astype(int)
    result["lm_bin"]      = result["Landmarks"].eq("Yes").astype(int)
    result["road_lanes"]  = result["road_enc"] * numeric_lanes

    result["is_peak"]      = (
        result["hour"].between(6, 9) | result["hour"].between(17, 21)
    ).astype(int)
    result["is_night"]     = (result["hour"] <= 5).astype(int)
    result["is_morning"]   = result["hour"].between(6, 11).astype(int)
    result["is_afternoon"] = result["hour"].between(12, 16).astype(int)
    result["is_evening"]   = result["hour"].between(17, 23).astype(int)

    return result


def oof_encode(train_df, test_df, col,
               target='demand', n_splits=5, smoothing=10):
    gm     = train_df[target].mean()
    enc_tr = np.full(len(train_df), gm)
    kf     = KFold(n_splits=n_splits, shuffle=False)

    for tr_i, val_i in kf.split(train_df):
        stats  = train_df.iloc[tr_i].groupby(col)[target].agg(['mean', 'count'])
        smooth = (
            stats['count'] * stats['mean'] + smoothing * gm
        ) / (stats['count'] + smoothing)
        enc_tr[val_i] = (
            train_df.iloc[val_i][col].map(smooth).fillna(gm).values
        )

    stats_f  = train_df.groupby(col)[target].agg(['mean', 'count'])
    smooth_f = (
        stats_f['count'] * stats_f['mean'] + smoothing * gm
    ) / (stats_f['count'] + smoothing)
    enc_te   = test_df[col].map(smooth_f).fillna(gm).values
    return enc_tr, enc_te


def full_pipeline(train_df, test_df):

    # Step 1 — geohash decode
    for df in [train_df, test_df]:
        coords     = df['geohash'].apply(lambda g: pgh.decode(g))
        df['lat']  = coords.apply(lambda x: x[0])
        df['lng']  = coords.apply(lambda x: x[1])
        df['geo3'] = df['geohash'].str[:3]
        df['geo4'] = df['geohash'].str[:4]
        df['geo5'] = df['geohash'].str[:5]

    # Step 2 — engineer features
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    # Step 3 — temperature imputation (train stats only)
    temp_map  = train_df.groupby('geohash')['Temperature'].median()
    temp_map3 = train_df.groupby('geo3')['Temperature'].median()
    global_t  = train_df['Temperature'].median()
    for df in [train_df, test_df]:
        df['Temperature'] = (
            df['Temperature']
            .fillna(df['geohash'].map(temp_map))
            .fillna(df['geo3'].map(temp_map3))
            .fillna(global_t)
        )

    # Step 4 — interaction keys
    for df in [train_df, test_df]:
        df['geo_slot']  = (
            df['geohash'].astype(str) + '_' + df['time_slot'].astype(str)
        )
        df['geo_peak']  = (
            df['geohash'].astype(str) + '_' + df['is_peak'].astype(str)
        )
        df['geo_road']  = (
            df['geohash'].astype(str) + '_' + df['road_enc'].astype(str)
        )

    # Step 5 — OOF target encoding
    encode_cols = [
        ('geohash',                  10),
        ('geo3',                     15),
        ('geo4',                     12),
        ('geo5',                     10),
        ('geo_slot',                  5),
        ('geo_peak',                  8),
        ('geo_road',                 10),
        ('road_weather_interaction',  8),
        ('geohash_hour_interaction',  5),
        ('geohash_weather',           6),
    ]
    for col, sm in encode_cols:
        train_df[f'{col}_enc'], test_df[f'{col}_enc'] = oof_encode(
            train_df, test_df, col, smoothing=sm
        )

    # Step 6 — geohash demand stats (train only → no leakage)
    geo_std  = train_df.groupby('geohash')['demand'].std()
    geo_max  = train_df.groupby('geohash')['demand'].max()
    geo_min  = train_df.groupby('geohash')['demand'].min()
    geo_mean = train_df.groupby('geohash')['demand'].mean()
    for df in [train_df, test_df]:
        df['geo_demand_std']   = df['geohash'].map(geo_std).fillna(0)
        df['geo_demand_max']   = df['geohash'].map(geo_max).fillna(0)
        df['geo_demand_min']   = df['geohash'].map(geo_min).fillna(0)
        df['geo_demand_range'] = df['geo_demand_max'] - df['geo_demand_min']
        df['geo_demand_mean']  = df['geohash'].map(geo_mean).fillna(0)

    # Step 7 — slot-level stats
    slot_mean = train_df.groupby('time_slot')['demand'].mean()
    slot_std  = train_df.groupby('time_slot')['demand'].std()
    for df in [train_df, test_df]:
        df['slot_demand_mean'] = df['time_slot'].map(slot_mean).fillna(0)
        df['slot_demand_std']  = df['time_slot'].map(slot_std).fillna(0)

    # Step 8 — geo×slot×day_of_week mean (most precise test lag)
    geo_slot_mean     = train_df.groupby(
        ['geohash', 'time_slot'])['demand'].mean()
    geo_slot_dow_mean = train_df.groupby(
        ['geohash', 'time_slot', 'day_of_week'])['demand'].mean()

    # Step 9 — lag features for TRAIN
    train_df = train_df.sort_values(
        ['geohash', 'day', 'time_slot']).reset_index(drop=True)

    train_df['demand_lag1']  = train_df.groupby('geohash')['demand'].shift(1)
    train_df['demand_lag4']  = train_df.groupby('geohash')['demand'].shift(4)
    train_df['demand_lag8']  = train_df.groupby('geohash')['demand'].shift(8)
    train_df['demand_roll4'] = train_df.groupby('geohash')['demand'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    train_df['demand_roll8'] = train_df.groupby('geohash')['demand'].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    train_df['lag1_vs_mean'] = (
        train_df['demand_lag1'] - train_df['geohash_enc']
    )

    # Step 10 — test lags: geo×slot×dow → geo×slot → geohash_enc fallback
    def get_test_lag(row):
        key3 = (row['geohash'], row['time_slot'], row['day_of_week'])
        key2 = (row['geohash'], row['time_slot'])
        if key3 in geo_slot_dow_mean.index:
            return geo_slot_dow_mean[key3]
        elif key2 in geo_slot_mean.index:
            return geo_slot_mean[key2]
        else:
            return row['geohash_enc']

    test_df['demand_lag1']  = test_df.apply(get_test_lag, axis=1)
    test_df['demand_lag4']  = test_df['demand_lag1']
    test_df['demand_lag8']  = test_df['demand_lag1']
    test_df['demand_roll4'] = test_df['demand_lag1']
    test_df['demand_roll8'] = test_df['demand_lag1']
    test_df['lag1_vs_mean'] = test_df['demand_lag1'] - test_df['geohash_enc']

    lag_cols = [
        'demand_lag1', 'demand_lag4', 'demand_lag8',
        'demand_roll4', 'demand_roll8', 'lag1_vs_mean'
    ]
    for col in lag_cols:
        train_df[col] = train_df[col].fillna(train_df['geohash_enc'])
        test_df[col]  = test_df[col].fillna(test_df['geohash_enc'])

    return train_df, test_df


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD, PREPROCESS, SPLIT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 3 — LOADING & PREPROCESSING")
print("="*60)

train = pd.read_csv("dataset/train.csv")
test  = pd.read_csv("dataset/test.csv")
print(f"  Raw train : {train.shape} | Raw test : {test.shape}")

# Run EDA before any processing
run_eda(train)

# Run pipeline
train, test = full_pipeline(train, test)

FEATURES = [
    # Spatial
    'lat', 'lng',
    'geohash_enc', 'geo3_enc', 'geo4_enc', 'geo5_enc',

    # Time
    'day', 'time_slot', 'hour', 'minute',
    'minutes_since_midnight',
    'hour_sin', 'hour_cos',
    'minute_sin', 'minute_cos',
    'day_sin', 'day_cos',
    'day_of_week', 'is_weekend',
    'is_peak', 'is_night', 'is_morning', 'is_afternoon', 'is_evening',

    # Road
    'road_enc', 'NumberofLanes', 'lv_bin', 'lm_bin',
    'road_lanes', 'capacity_proxy',

    # Weather
    'weather_enc', 'Temperature',
    'temp_weather_interaction', 'lanes_weather',

    # OOF interactions
    'geo_slot_enc', 'geo_peak_enc', 'geo_road_enc',
    'road_weather_interaction_enc',
    'geohash_hour_interaction_enc',
    'geohash_weather_enc',

    # Demand stats
    'geo_demand_std', 'geo_demand_max',
    'geo_demand_min', 'geo_demand_range', 'geo_demand_mean',
    'slot_demand_mean', 'slot_demand_std',

    # Lag
    'demand_lag1', 'demand_lag4', 'demand_lag8',
    'demand_roll4', 'demand_roll8', 'lag1_vs_mean',
]

X      = train[FEATURES]
y      = train['demand']
X_test = test[FEATURES]

print(f"\n  Total features : {len(FEATURES)}")
print(f"  Train          : {X.shape}")
print(f"  Test           : {X_test.shape}")
print(f"  Missing train  : {X.isnull().sum().sum()}")
print(f"  Missing test   : {X_test.isnull().sum().sum()}")

# ── Day-based split (honest CV matching leaderboard) ──────────────────
train_sorted = train.sort_values(['day', 'time_slot']).reset_index(drop=True)
unique_days  = sorted(train_sorted['day'].unique())
val_day      = unique_days[-1]
train_days   = unique_days[:-1]

train_mask = train_sorted['day'].isin(train_days)
val_mask   = train_sorted['day'] == val_day

X_tr  = train_sorted[FEATURES][train_mask]
y_tr  = train_sorted['demand'][train_mask]
X_val = train_sorted[FEATURES][val_mask]
y_val = train_sorted['demand'][val_mask]

print(f"\n  Train days : {train_days} → {len(X_tr):,} rows")
print(f"  Val day    : {val_day}  → {len(X_val):,} rows")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — OPTUNA TUNING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 4 — HYPERPARAMETER TUNING")
print("="*60)

# ── XGBoost ────────────────────────────────────────────────────────────
def xgb_objective(trial):
    params = {
        "n_estimators":          3000,
        "early_stopping_rounds": 50,
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "min_child_weight":  trial.suggest_int("min_child_weight", 2, 20),
        "max_delta_step":    trial.suggest_int("max_delta_step", 0, 10),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode":  trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "random_state": 42, "device": "cuda",
        "tree_method": "hist", "verbosity": 0,
    }
    m = XGBRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return max(0, 100 * r2_score(y_val, m.predict(X_val)))

print("\n  Tuning XGBoost (50 trials)...")
xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
print(f"  XGBoost best val score : {xgb_study.best_value:.2f}")

# ── LightGBM ───────────────────────────────────────────────────────────
def lgb_objective(trial):
    params = {
        "n_estimators":      3000,
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "random_state": 42, "device": "gpu", "verbose": -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return max(0, 100 * r2_score(y_val, m.predict(X_val)))

print("\n  Tuning LightGBM (50 trials)...")
lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)
print(f"  LightGBM best val score : {lgb_study.best_value:.2f}")

# ── CatBoost ───────────────────────────────────────────────────────────
def cat_objective(trial):
    params = {
        "iterations":            3000,
        "early_stopping_rounds": 50,
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "depth":             trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_seed": 42, "task_type": "GPU", "verbose": 0,
    }
    m = CatBoostRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    return max(0, 100 * r2_score(y_val, m.predict(X_val)))

print("\n  Tuning CatBoost (50 trials)...")
cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=50, show_progress_bar=True)
print(f"  CatBoost best val score : {cat_study.best_value:.2f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — RETRAIN ON FULL DATA
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 5 — RETRAINING ON FULL DATA")
print("="*60)

# XGBoost
xgb_params = xgb_study.best_trial.params
xgb_params.update({
    "n_estimators": 3000, "random_state": 42,
    "device": "cuda", "tree_method": "hist", "verbosity": 0,
})
xgb_params.pop("early_stopping_rounds", None)
best_xgb = XGBRegressor(**xgb_params)
best_xgb.fit(X, y)
print("  ✅ XGBoost retrained")

# LightGBM
lgb_params = lgb_study.best_trial.params
lgb_params.update({
    "n_estimators": 3000, "random_state": 42,
    "device": "gpu", "subsample_freq": 1, "verbose": -1,
})
best_lgb = lgb.LGBMRegressor(**lgb_params)
best_lgb.fit(X, y)
print("  ✅ LightGBM retrained")

# CatBoost
cat_params = cat_study.best_trial.params
cat_params.update({
    "iterations": 3000, "random_seed": 42,
    "task_type": "GPU", "verbose": 0,
})
cat_params.pop("early_stopping_rounds", None)
best_cat = CatBoostRegressor(**cat_params)
best_cat.fit(X, y, verbose=False)
print("  ✅ CatBoost retrained")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — SCORE CHECK & ENSEMBLE WEIGHTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 6 — SCORE CHECK")
print("="*60)

xgb_val_preds = best_xgb.predict(X_val)
lgb_val_preds = best_lgb.predict(X_val)
cat_val_preds = best_cat.predict(X_val)

xgb_val_score = max(0, 100 * r2_score(y_val, xgb_val_preds))
lgb_val_score = max(0, 100 * r2_score(y_val, lgb_val_preds))
cat_val_score = max(0, 100 * r2_score(y_val, cat_val_preds))

total = xgb_val_score + lgb_val_score + cat_val_score
xgb_w = xgb_val_score / total
lgb_w = lgb_val_score / total
cat_w = cat_val_score / total

ensemble_val   = (xgb_w * xgb_val_preds +
                  lgb_w * lgb_val_preds +
                  cat_w * cat_val_preds)
ensemble_score = max(0, 100 * r2_score(y_val, ensemble_val))

print(f"\n  XGBoost  Val Score : {xgb_val_score:.2f} / 100")
print(f"  LightGBM Val Score : {lgb_val_score:.2f} / 100")
print(f"  CatBoost Val Score : {cat_val_score:.2f} / 100")
print(f"\n  Weights → XGB:{xgb_w:.2f}  LGB:{lgb_w:.2f}  CAT:{cat_w:.2f}")
print(f"  Ensemble Val Score : {ensemble_score:.2f} / 100")
print(f"\n  ⚠ If this ≈ leaderboard score → CV is honest")

# Train score to check gap
train_ens  = (xgb_w * best_xgb.predict(X) +
              lgb_w * best_lgb.predict(X) +
              cat_w * best_cat.predict(X))
train_score = max(0, 100 * r2_score(y, train_ens))
gap         = train_score - ensemble_score
print(f"\n  Train Score : {train_score:.2f} / 100")
print(f"  Gap         : {gap:.2f}  "
      f"{'✅ Healthy' if gap < 5 else '⚠ Mild' if gap < 10 else '❌ Overfit'}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — FEATURE IMPORTANCE PLOT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 7 — FEATURE IMPORTANCE")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle('Feature Importance — All Models', fontsize=16, fontweight='bold')

# XGBoost
xgb_imp = pd.Series(best_xgb.feature_importances_,
                     index=FEATURES).sort_values(ascending=True).tail(20)
axes[0].barh(xgb_imp.index, xgb_imp.values, color=PALETTE[0], alpha=0.85)
axes[0].set_title('XGBoost — Top 20')
axes[0].set_xlabel('Importance')

# LightGBM
lgb_imp = pd.Series(best_lgb.feature_importances_,
                     index=FEATURES).sort_values(ascending=True).tail(20)
axes[1].barh(lgb_imp.index, lgb_imp.values, color=PALETTE[1], alpha=0.85)
axes[1].set_title('LightGBM — Top 20')
axes[1].set_xlabel('Importance')

# CatBoost
cat_imp = pd.Series(best_cat.get_feature_importance(),
                     index=FEATURES).sort_values(ascending=True).tail(20)
axes[2].barh(cat_imp.index, cat_imp.values, color=PALETTE[2], alpha=0.85)
axes[2].set_title('CatBoost — Top 20')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('plot7_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  ✅ plot7_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — FIRST SUBMISSION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 8 — FIRST SUBMISSION")
print("="*60)

xgb_test  = best_xgb.predict(X_test)
lgb_test  = best_lgb.predict(X_test)
cat_test  = best_cat.predict(X_test)

final_preds = np.clip(
    xgb_w * xgb_test + lgb_w * lgb_test + cat_w * cat_test,
    0, 1
)

submission = pd.DataFrame({
    "Index":  test["Index"],
    "demand": final_preds
})
submission.to_csv("submission.csv", index=False)
print(f"  ✅ submission.csv saved  shape={submission.shape}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — PSEUDO LABELING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 9 — PSEUDO LABELING")
print("="*60)

test_pseudo           = test.copy()
test_pseudo['demand'] = final_preds

# Only use high-confidence pseudo labels (middle range, not extremes)
confident_mask = (final_preds > 0.05) & (final_preds < 0.95)
test_pseudo    = test_pseudo[confident_mask].reset_index(drop=True)
print(f"  Pseudo labeled rows : {len(test_pseudo):,} / {len(test):,}")

train_ext = pd.concat([train, test_pseudo], ignore_index=True)
X_ext     = train_ext[FEATURES]
y_ext     = train_ext['demand']

# Retrain all 3 on extended data
best_xgb_v2 = XGBRegressor(**xgb_params)
best_xgb_v2.fit(X_ext, y_ext)

best_lgb_v2 = lgb.LGBMRegressor(**lgb_params)
best_lgb_v2.fit(X_ext, y_ext)

best_cat_v2 = CatBoostRegressor(**cat_params)
best_cat_v2.fit(X_ext, y_ext, verbose=False)

# Val scores on pseudo-trained models
xgb_v2_score = max(0, 100 * r2_score(y_val, best_xgb_v2.predict(X_val)))
lgb_v2_score = max(0, 100 * r2_score(y_val, best_lgb_v2.predict(X_val)))
cat_v2_score = max(0, 100 * r2_score(y_val, best_cat_v2.predict(X_val)))

total_v2 = xgb_v2_score + lgb_v2_score + cat_v2_score
xgb_w2   = xgb_v2_score / total_v2
lgb_w2   = lgb_v2_score / total_v2
cat_w2   = cat_v2_score / total_v2

ens_v2_val = (xgb_w2 * best_xgb_v2.predict(X_val) +
              lgb_w2 * best_lgb_v2.predict(X_val) +
              cat_w2 * best_cat_v2.predict(X_val))
ens_v2_score = max(0, 100 * r2_score(y_val, ens_v2_val))

print(f"\n  After pseudo labeling:")
print(f"  XGBoost  : {xgb_v2_score:.2f}  LightGBM : {lgb_v2_score:.2f}  CatBoost : {cat_v2_score:.2f}")
print(f"  Ensemble : {ens_v2_score:.2f} / 100")
print(f"  Change   : {ens_v2_score - ensemble_score:+.2f} pts")

# Use pseudo model only if it's better
if ens_v2_score > ensemble_score:
    print("  ✅ Pseudo labeling improved score — using v2 predictions")
    final_preds_v2 = np.clip(
        xgb_w2 * best_xgb_v2.predict(X_test) +
        lgb_w2 * best_lgb_v2.predict(X_test) +
        cat_w2 * best_cat_v2.predict(X_test),
        0, 1
    )
else:
    print("  ⚠ Pseudo labeling did not improve — keeping original predictions")
    final_preds_v2 = final_preds

submission_v2 = pd.DataFrame({
    "Index":  test["Index"],
    "demand": final_preds_v2
})
submission_v2.to_csv("submission_pseudo.csv", index=False)
print(f"  ✅ submission_pseudo.csv saved  shape={submission_v2.shape}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — FINAL SUMMARY PLOT
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')

models  = ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble', 'Ensemble+PL']
scores  = [xgb_val_score, lgb_val_score, cat_val_score,
           ensemble_score, ens_v2_score]
colors  = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4]]

bars = axes[0].bar(models, scores, color=colors, alpha=0.85)
axes[0].set_ylim(min(scores) - 2, 100)
axes[0].set_title('Val Scores by Model')
axes[0].set_ylabel('Score / 100')
for bar, score in zip(bars, scores):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f'{score:.1f}', ha='center', va='bottom', fontsize=9)

# Prediction distribution
axes[1].hist(final_preds,    bins=50, alpha=0.6,
             color=PALETTE[0], label='Original')
axes[1].hist(final_preds_v2, bins=50, alpha=0.6,
             color=PALETTE[1], label='Pseudo Label')
axes[1].set_title('Prediction Distribution')
axes[1].set_xlabel('Predicted Demand')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('plot8_model_summary.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n  ✅ plot8_model_summary.png")

print("\n" + "="*60)
print("  DONE")
print(f"  Best val score    : {max(ensemble_score, ens_v2_score):.2f} / 100")
print(f"  Submit first      : submission.csv")
print(f"  Submit second     : submission_pseudo.csv")
print("="*60)