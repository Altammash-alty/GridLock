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


