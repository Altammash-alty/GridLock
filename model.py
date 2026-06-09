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