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
