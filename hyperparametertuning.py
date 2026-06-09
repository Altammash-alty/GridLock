
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
