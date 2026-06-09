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
