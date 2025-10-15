#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Honest global imputation + fair evaluation (fixed 14d + random 14d gaps).

- Trains ONE model per pollutant on observed rows (time features + other pollutants; NO lags).
- Uses that SAME trained model to:
    * Impute real missing values.
    * Evaluate on a fixed 14-day block (no refit).
    * Evaluate on K random 14-day blocks (no refit) and summarize.
- Outputs:
    data/train_imputed_all_columns.csv
    validation/<col>_honest_eval.csv
    validation/<col>_fixed14_line.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- CONFIG ----------------
INPUT_FILE     = "/home/rico/HEC/07-time_series/paris-air-quality-prediction/data/train_interpolated_small_gaps.csv"
OUTPUT_FILE    = "data/train_imputed_all_columns.csv"
VALIDATION_DIR = "validation"
DT_COL         = "id"
TARGETS        = ["valeur_CO", "valeur_NO2", "valeur_O3", "valeur_PM10", "valeur_PM25"]

# Evaluation windows
GAP_HOURS      = 14 * 24
BUFFER_HOURS   = 24                # buffer inside observed run for the fixed gap
RAND_GAPS_K    = 5                 # number of random gap samples per target
RANDOM_SEED    = 42
# ---------------------------------------


def log(s: str):
    print(s, flush=True)


def ensure_dirs():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)


def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[dt_col], errors="coerce")
    out["_hour"]  = ts.dt.hour
    out["_dow"]   = ts.dt.weekday
    out["_month"] = ts.dt.month
    for name, maxv in [("_hour", 24), ("_dow", 7), ("_month", 12)]:
        s = out[name].astype(float)
        out[f"{name}_sin"] = np.sin(2*np.pi*s/maxv)
        out[f"{name}_cos"] = np.cos(2*np.pi*s/maxv)
    return out


def longest_consecutive_run(indices: np.ndarray) -> Tuple[int, int, int]:
    if len(indices) == 0:
        return None, None, 0
    best_s = indices[0]; best_e = indices[0]; best_len = 1
    cur_s  = indices[0]; cur_e  = indices[0]; cur_len = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            cur_e = indices[i]; cur_len += 1
        else:
            if cur_len > best_len:
                best_s, best_e, best_len = cur_s, cur_e, cur_len
            cur_s = indices[i]; cur_e = indices[i]; cur_len = 1
    if cur_len > best_len:
        best_s, best_e, best_len = cur_s, cur_e, cur_len
    return best_s, best_e, best_len


def build_pipeline(numeric_features: List[str], categorical_features: List[str], use_xgb: bool):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    if use_xgb:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=max(1, (os.cpu_count() or 2) - 1),
            random_state=RANDOM_SEED,
            tree_method="hist",
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.06,
            max_depth=None,
            random_state=RANDOM_SEED,
        )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def pick_fixed_gap_indices(y: pd.Series) -> Tuple[np.ndarray, bool]:
    """Return boolean mask for a fixed 14-day gap within the longest fully observed run."""
    obs_mask = y.notna().to_numpy()
    obs_idx  = np.where(obs_mask)[0]
    ok = False
    if len(obs_idx) > 0:
        s, e, L = longest_consecutive_run(obs_idx)
        if L >= GAP_HOURS + BUFFER_HOURS:
            hold_start = s + BUFFER_HOURS
            hold_end   = hold_start + GAP_HOURS - 1
            if hold_end <= e:
                m = np.zeros_like(obs_mask, dtype=bool)
                m[hold_start:hold_end+1] = True
                ok = True
                return m, ok
    return np.zeros_like(obs_mask, dtype=bool), ok


def sample_random_gaps(y: pd.Series, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Sample k random 14-day masks fully inside observed segments.
    Returns list of boolean masks.
    """
    masks = []
    obs = y.notna().to_numpy()
    idx = np.where(obs)[0]
    if len(idx) == 0:
        return masks

    # Build list of (start,end) observed segments
    segs = []
    s = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((s, prev))
            s = i; prev = i
    segs.append((s, prev))

    # Filter segments long enough for GAP_HOURS
    segs = [(a,b) for a,b in segs if (b - a + 1) >= GAP_HOURS]
    if not segs:
        return masks

    for _ in range(k):
        a,b = segs[rng.integers(0, len(segs))]
        start = rng.integers(a, b - GAP_HOURS + 2)  # inclusive
        end = start + GAP_HOURS - 1
        m = np.zeros_like(obs, dtype=bool)
        m[start:end+1] = True
        masks.append(m)
    return masks


def honest_eval_same_model(pipe: Pipeline, X_all: pd.DataFrame, y_all: pd.Series,
                           fixed_mask: np.ndarray, random_masks: List[np.ndarray],
                           time_index: pd.Series, target: str, base_path: str):
    """
    Evaluate using the SAME trained model:
     - fixed 14-day mask
     - random K masks
    Writes CSV with metrics and a PNG for the fixed mask plot.
    """
    rows = []

    def compute_scores(mask: np.ndarray, tag: str):
        te_idx = np.where(mask & y_all.notna().to_numpy())[0]
        if len(te_idx) == 0:
            return {"tag": tag, "note": "no_test_rows"}
        y_te = y_all.iloc[te_idx]
        X_te = X_all.iloc[te_idx]
        y_hat = pipe.predict(X_te)
        mae = float(mean_absolute_error(y_te, y_hat))
        r2  = float(r2_score(y_te, y_hat))
        thr = float(y_all.dropna().quantile(0.9))
        y_hat_s = pd.Series(y_hat, index=y_te.index)
        pred_peaks = int((y_hat_s >= thr).sum())
        true_peaks = int((y_te >= thr).sum())
        both_peaks = int(((y_hat_s >= thr) & (y_te >= thr)).sum())
        precision  = float(both_peaks / pred_peaks) if pred_peaks > 0 else float("nan")
        recall     = float(both_peaks / true_peaks) if true_peaks > 0 else float("nan")
        return {
            "tag": tag,
            "mae": mae, "r2": r2,
            "peak_threshold": thr,
            "pred_peak_count": pred_peaks,
            "true_peak_count": true_peaks,
            "peak_precision": precision,
            "peak_recall": recall,
            "n_test": int(len(te_idx))
        }

    # Fixed mask
    fixed_scores = compute_scores(fixed_mask, "fixed14")
    rows.append(fixed_scores)

    # Plot for fixed gap (only if it produced test rows)
    if fixed_scores.get("n_test", 0) > 0:
        te_idx = np.where(fixed_mask & y_all.notna().to_numpy())[0]
        t = time_index.iloc[te_idx]
        y_te = y_all.iloc[te_idx]
        y_hat = pipe.predict(X_all.iloc[te_idx])
        fig = plt.figure()
        plt.plot(t, y_te.values, label="True")
        plt.plot(t, y_hat, label="Pred")
        plt.title(f"Fixed 14-day Gap — {target} (honest, same model)")
        plt.xlabel("Time"); plt.ylabel(target)
        plt.legend(); plt.tight_layout()
        plot_path = os.path.join(base_path, f"{target}_fixed14_line.png")
        plt.savefig(plot_path, dpi=150); plt.close(fig)

    # Random masks
    rand_scores = []
    for i, m in enumerate(random_masks, 1):
        sc = compute_scores(m, f"random14_{i}")
        rows.append(sc)
        if sc.get("n_test", 0) > 0:
            rand_scores.append(sc)

    # Summaries for random
    if rand_scores:
        mae_vals = [r["mae"] for r in rand_scores if "mae" in r]
        r2_vals  = [r["r2"]  for r in rand_scores if "r2"  in r]
        rows.append({
            "tag": "random14_summary",
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "r2_mean": float(np.mean(r2_vals)),
            "r2_std": float(np.std(r2_vals)),
            "n_samples": len(rand_scores)
        })
    return pd.DataFrame(rows)


def main():
    ensure_dirs()

    if not os.path.exists(INPUT_FILE):
        log(f"❌ File not found: {INPUT_FILE}")
        sys.exit(1)

    log(f"🔄 Loading {os.path.abspath(INPUT_FILE)} ...")
    df = pd.read_csv(INPUT_FILE)
    if DT_COL not in df.columns:
        log(f"❌ datetime column '{DT_COL}' not found in CSV.")
        sys.exit(1)

    # Sort by time
    df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
    df = df.sort_values(DT_COL).reset_index(drop=True)

    # Only keep targets that exist
    targets = [c for c in TARGETS if c in df.columns]
    if not targets:
        log("❌ None of the target columns found in CSV.")
        sys.exit(1)

    # Add time features once
    df_feat = add_time_features(df, DT_COL)
    time_cols = [c for c in df_feat.columns if c.startswith("_hour") or c.startswith("_dow") or c.startswith("_month")]

    # Split feature types for pipelines
    try:
        import xgboost as xgb  # noqa
        use_xgb_global = True
    except Exception:
        use_xgb_global = False

    # This will accumulate imputed columns
    df_out = df_feat.copy()

    rng = np.random.default_rng(RANDOM_SEED)

    for target in targets:
        log(f"\n🧩 Fitting global imputation model for: {target}")

        # Features = other pollutants + time features (no lags)
        other_pollutants = [c for c in targets if c != target and c in df_feat.columns]
        feature_cols = other_pollutants + time_cols

        X_full = df_feat[feature_cols].copy()
        y_full = df_feat[target].astype(float)

        num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_full.select_dtypes(include=["object","category","bool"]).columns.tolist()

        obs_mask = y_full.notna()
        if obs_mask.sum() == 0:
            log(f"⚠️ No observed data for {target}; skipping.")
            continue

        pipe = build_pipeline(num_cols, cat_cols, use_xgb_global)
        pipe.fit(X_full.loc[obs_mask], y_full.loc[obs_mask])

        # ---- Honest evaluation using SAME trained model ----
        # Build masks on observed rows only
        fixed_mask, ok_fixed = pick_fixed_gap_indices(y_full)
        rand_masks = sample_random_gaps(y_full, RAND_GAPS_K, rng)

        # Prepare X_all without the target; pipeline handles missing features
        metrics_df = honest_eval_same_model(
            pipe=pipe,
            X_all=X_full,
            y_all=y_full,
            fixed_mask=fixed_mask,
            random_masks=rand_masks,
            time_index=df[DT_COL],
            target=target,
            base_path=VALIDATION_DIR,
        )
        metrics_path = os.path.join(VALIDATION_DIR, f"{target}_honest_eval.csv")
        metrics_df.to_csv(metrics_path, index=False)
        if ok_fixed:
            log(f"   📊 saved metrics → {metrics_path}")
            log(f"   (fixed gap tag: 'fixed14', random gap tags: 'random14_*', plus summary)")

        # ---- Impute real missing values with the SAME model ----
        miss_mask = y_full.isna()
        if miss_mask.any():
            y_pred = pipe.predict(X_full.loc[miss_mask])
            df_out.loc[miss_mask, target] = y_pred
        # Keep observed as-is
        df_out.loc[obs_mask, target] = y_full.loc[obs_mask]

    # Drop engineered time features from final output (keep original schema)
    drop_cols = [c for c in df_out.columns if c.startswith("_hour") or c.startswith("_dow") or c.startswith("_month")]
    final_df = df_out.drop(columns=drop_cols)

    final_df.to_csv(OUTPUT_FILE, index=False)
    log(f"\n💾 Wrote merged imputed file: {OUTPUT_FILE}")
    log("✅ Done.")
    

if __name__ == "__main__":
    main()
