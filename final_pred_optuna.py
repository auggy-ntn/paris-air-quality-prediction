#!/usr/bin/env python3
# final_pred_lags_2.py
# TFT with lags + (optional) quick hyperparameter search.
# Keeps your rolling forecast (with delta shrink + caps + light EMA) intact.

import os
import warnings
from dataclasses import dataclass
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping

# ---------- NEW: Optuna (optional) ----------
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# make float32 the default everywhere
torch.set_default_dtype(torch.float32)

# ======================= CONFIG =======================
DT_COL = "id"
TARGETS = ["valeur_CO", "valeur_NO2", "valeur_O3", "valeur_PM10", "valeur_PM25"]

# lags (in hours) to use as past covariates from the targets
LAGS = (1, 24, 168)   # 1h, 1d, 1w

# context / horizon
INPUT_CHUNK_LEN = 336     # ~2 weeks of history
OUTPUT_CHUNK_LEN = 24     # 1 day
FORECAST_STEPS = 504      # required prediction length

# training
VAL_FRAC = 0.10
MAX_EPOCHS = 1            # (final stage) increase later for better results
BATCH_SIZE = 64
SEED = 42
USE_GPU = torch.cuda.is_available()

# ---------- NEW: quick tuning controls ----------
USE_TUNING = True          # set False to skip tuning
N_TRIALS = 5               # quick search; bump to e.g. 20–50 later
MAX_EPOCHS_TUNE = 1        # keep tiny to be fast; can try 2–3 later

# I/O
TRAIN_CSV = "data/train_imputed_all_columns.csv"
TEST_CSV  = "data/test.csv"
OUT_PATH  = "submissions/submission_tft_lags_2.csv"

# safety / post-process
CLIP_NEGATIVES = True
TIMESTAMP_FMT = "%Y-%m-%d %H"

# ----- Smoothness controls (penalize hour-to-hour jumps) -----
DELTA_CAP_Q      = 0.95  # cap abs(delta) to 95th percentile of train deltas
DELTA_SHRINK     = 0.50  # multiply predicted deltas by this (0.0..1.0)
EMA_SMOOTH_ALPHA = 0.20  # blend with last level: new = alpha*last + (1-alpha)*new

# ======================================================

def set_repro(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(train_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    train_df[DT_COL] = pd.to_datetime(train_df[DT_COL], errors="raise")
    test_df[DT_COL]  = pd.to_datetime(test_df[DT_COL],  errors="raise")
    train_df = train_df.sort_values(DT_COL).reset_index(drop=True)
    test_df  = test_df.sort_values(DT_COL).reset_index(drop=True)
    return train_df, test_df

def to_multivar_series(df: pd.DataFrame, cols: Sequence[str]) -> TimeSeries:
    ts = TimeSeries.from_dataframe(
        df, time_col=DT_COL, value_cols=list(cols),
        fill_missing_dates=True, freq="h"
    )
    return ts.astype(np.float32)

def build_future_covs(train_idx: pd.Series,
                      test_idx: pd.Series) -> Tuple[TimeSeries, TimeSeries]:
    t_train_min = pd.to_datetime(train_idx.min())
    t_train_max = pd.to_datetime(train_idx.max())
    t_test_max  = pd.to_datetime(test_idx.max())

    full_end = t_test_max + pd.Timedelta(hours=OUTPUT_CHUNK_LEN - 1)
    full_times = pd.date_range(start=t_train_min, end=full_end, freq="h")
    train_times = pd.date_range(start=t_train_min, end=t_train_max, freq="h")

    def make_feat(times: pd.DatetimeIndex) -> pd.DataFrame:
        f = pd.DataFrame({DT_COL: times})
        f["hour"] = f[DT_COL].dt.hour
        f["dow"]  = f[DT_COL].dt.dayofweek
        f["month"]= f[DT_COL].dt.month
        f["doy"]  = f[DT_COL].dt.dayofyear
        f["weekend"] = (f["dow"]>=5).astype(np.float32)
        f["sin_hour"] = np.sin(2*np.pi*f["hour"]/24.0)
        f["cos_hour"] = np.cos(2*np.pi*f["hour"]/24.0)
        f["sin_dow"]  = np.sin(2*np.pi*f["dow"]/7.0)
        f["cos_dow"]  = np.cos(2*np.pi*f["dow"]/7.0)
        f["sin_doy"]  = np.sin(2*np.pi*f["doy"]/365.25)
        f["cos_doy"]  = np.cos(2*np.pi*f["doy"]/365.25)
        return f[[DT_COL,"weekend","sin_hour","cos_hour","sin_dow","cos_dow","sin_doy","cos_doy"]]

    ft_train_df = make_feat(train_times)
    ft_full_df  = make_feat(full_times)

    ft_train = TimeSeries.from_dataframe(
        ft_train_df, time_col=DT_COL,
        value_cols=[c for c in ft_train_df.columns if c!=DT_COL],
        fill_missing_dates=True, freq="h"
    ).astype(np.float32)

    ft_all   = TimeSeries.from_dataframe(
        ft_full_df, time_col=DT_COL,
        value_cols=[c for c in ft_full_df.columns if c!=DT_COL],
        fill_missing_dates=True, freq="h"
    ).astype(np.float32)

    return ft_train, ft_all

def build_target_lag_covs(train_df: pd.DataFrame,
                          targets: Sequence[str],
                          lags: Sequence[int]) -> Tuple[TimeSeries, pd.DataFrame]:
    df = train_df[[DT_COL] + list(targets)].copy()
    df = df.sort_values(DT_COL).reset_index(drop=True)
    for m in targets:
        for L in lags:
            df[f"{m}_lag{L}"] = df[m].shift(L)
    trim = max(lags)
    df_aligned = df.iloc[trim:].reset_index(drop=True)
    lag_cols = [c for c in df_aligned.columns if any(c.endswith(f"_lag{L}") for L in lags)]
    pc = TimeSeries.from_dataframe(
        df_aligned, time_col=DT_COL, value_cols=lag_cols,
        fill_missing_dates=True, freq="h"
    ).astype(np.float32)
    return pc, df_aligned[[DT_COL] + list(targets)]

def split_train_val(ts: TimeSeries, frac: float) -> Tuple[TimeSeries, TimeSeries]:
    split_idx = int(len(ts) * (1.0 - frac))
    return ts[:split_idx], ts[split_idx:]

def compute_delta_caps(df_aligned_targets: pd.DataFrame,
                       targets: Sequence[str],
                       q: float) -> dict:
    caps = {}
    g = df_aligned_targets.sort_values(DT_COL).reset_index(drop=True)
    for col in targets:
        d = g[col].astype(float).diff().abs().dropna()
        caps[col] = float(d.quantile(q)) if len(d) else np.inf
    return caps    

@dataclass
class ScaledData:
    y_all_s: TimeSeries
    y_scaler: Scaler
    pc_train_s: TimeSeries
    pc_scaler: Scaler
    ft_train_s: TimeSeries
    ft_all_s: TimeSeries
    ft_scaler: Scaler

def scale_all(y_all: TimeSeries, pc_train: TimeSeries,
              ft_train: TimeSeries, ft_all: TimeSeries) -> ScaledData:
    y_scaler = Scaler()
    y_all_s = y_scaler.fit_transform(y_all).astype(np.float32)
    pc_scaler = Scaler()
    pc_train_s = pc_scaler.fit_transform(pc_train).astype(np.float32)
    ft_scaler = Scaler()
    ft_train_s = ft_scaler.fit_transform(ft_train).astype(np.float32)
    ft_all_s   = ft_scaler.transform(ft_all).astype(np.float32)
    return ScaledData(y_all_s, y_scaler, pc_train_s, pc_scaler, ft_train_s, ft_all_s, ft_scaler)

def _pl_common_kwargs(enable_val: bool, max_epochs: int):
    # When no val set is passed, monitor train_loss to avoid Lightning errors.
    monitor_metric = "val_loss" if enable_val else "train_loss"
    return dict(
        accelerator="gpu" if USE_GPU else "cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=True,
        callbacks=[EarlyStopping(monitor=monitor_metric, mode="min", patience=3, verbose=True)],
        max_epochs=max_epochs,
        log_every_n_steps=50,
    )

def make_model_first_stage(hp=None) -> TFTModel:
    hp = hp or {}
    return TFTModel(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_size=hp.get("hidden_size", 40),
        lstm_layers=hp.get("lstm_layers", 3),
        dropout=hp.get("dropout", 0.2),
        num_attention_heads=hp.get("num_attention_heads", 1),
        batch_size=hp.get("batch_size", BATCH_SIZE),
        n_epochs=hp.get("n_epochs", MAX_EPOCHS_TUNE if USE_TUNING else MAX_EPOCHS),
        add_relative_index=True,
        random_state=SEED,
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        optimizer_kwargs={
            "lr": hp.get("lr", 1e-3),
            "weight_decay": hp.get("weight_decay", 0.0),
        },
        pl_trainer_kwargs=_pl_common_kwargs(enable_val=True, max_epochs=hp.get("n_epochs", MAX_EPOCHS_TUNE if USE_TUNING else MAX_EPOCHS)),
    )

def make_model_finetune(hp=None) -> TFTModel:
    hp = hp or {}
    return TFTModel(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_size=hp.get("hidden_size", 40),
        lstm_layers=hp.get("lstm_layers", 3),
        dropout=hp.get("dropout", 0.2),
        num_attention_heads=hp.get("num_attention_heads", 1),
        batch_size=hp.get("batch_size", BATCH_SIZE),
        n_epochs=hp.get("n_epochs", MAX_EPOCHS),
        add_relative_index=True,
        random_state=SEED,
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        optimizer_kwargs={
            "lr": hp.get("lr", 1e-3),
            "weight_decay": hp.get("weight_decay", 0.0),
        },
        pl_trainer_kwargs=_pl_common_kwargs(enable_val=False, max_epochs=hp.get("n_epochs", MAX_EPOCHS)),
    )

# -------- Lag window builder for prediction --------
def _build_pc_window_from_buffer(
    buf_idxed: pd.DataFrame,
    end_time: pd.Timestamp,
    targets: Sequence[str],
    lags: Sequence[int],
    window_len: int,
) -> TimeSeries:
    times = pd.date_range(end=end_time, periods=window_len, freq="h")
    lag_cols = [f"{m}_lag{L}" for m in targets for L in lags]
    arr = np.empty((window_len, len(lag_cols)), dtype=np.float32)
    col_i = 0
    for m in targets:
        series_m = buf_idxed[m]
        for L in lags:
            src_times = times - pd.to_timedelta(L, unit="h")
            vals = series_m.reindex(src_times).ffill()
            if vals.isna().any():
                vals = vals.fillna(series_m.iloc[-1])
            arr[:, col_i] = vals.values.astype(np.float32)
            col_i += 1
    pc_window = TimeSeries.from_times_and_values(times=times, values=arr, columns=lag_cols)
    return pc_window.astype(np.float32)

def rolling_predict_with_lags(
    model: TFTModel,
    y_hist_s: TimeSeries,
    ft_all_s: TimeSeries,
    work_df_unscaled: pd.DataFrame,   # columns: [id] + targets (unscaled)
    y_scaler: Scaler,
    pc_scaler: Scaler,
    targets: Sequence[str],
    lags: Sequence[int],
    steps: int,
    delta_caps: dict,                 # per-target delta cap
    delta_shrink: float = DELTA_SHRINK,
    ema_alpha: float = EMA_SMOOTH_ALPHA,
) -> TimeSeries:
    preds: List[TimeSeries] = []
    buf_idxed = work_df_unscaled.set_index(DT_COL).sort_index()
    for _ in range(steps):
        t_next = buf_idxed.index[-1] + pd.Timedelta(hours=1)
        enc_end = t_next - pd.Timedelta(hours=1)
        pc_window = _build_pc_window_from_buffer(
            buf_idxed=buf_idxed, end_time=enc_end, targets=targets, lags=lags,
            window_len=INPUT_CHUNK_LEN + 1,
        ).astype(np.float32)
        pc_window_s = pc_scaler.transform(pc_window).astype(np.float32)
        window_start = pc_window.start_time()
        dec_end = t_next + pd.Timedelta(hours=OUTPUT_CHUNK_LEN - 1)
        ft_window_s = ft_all_s.slice(window_start, dec_end).astype(np.float32)
        yhat_s = model.predict(
            n=1, series=y_hist_s,
            past_covariates=pc_window_s, future_covariates=ft_window_s,
            verbose=True,
        )
        yhat_lvl = y_scaler.inverse_transform(yhat_s)
        prev_lvl = buf_idxed.iloc[-1][targets].astype(float).values
        raw_pred = yhat_lvl.values(copy=False)[0]
        delta = raw_pred - prev_lvl
        delta = delta_shrink * delta
        for i, col in enumerate(targets):
            cap = delta_caps.get(col, np.inf)
            if np.isfinite(cap) and abs(delta[i]) > cap:
                delta[i] = np.sign(delta[i]) * cap
        penalized = prev_lvl + delta
        smoothed = ema_alpha * prev_lvl + (1.0 - ema_alpha) * penalized
        yhat_lvl_smoothed = TimeSeries.from_times_and_values(
            times=pd.date_range(start=t_next, periods=1, freq="h"),
            values=smoothed.reshape(1, -1).astype(np.float32),
            columns=targets,
        )
        preds.append(yhat_lvl_smoothed)
        buf_idxed = pd.concat([buf_idxed, pd.DataFrame({m: float(v) for m, v in zip(targets, smoothed)}, index=[t_next])])
        y_hist_s = y_hist_s.concatenate(y_scaler.transform(yhat_lvl_smoothed).astype(np.float32))
    pred_ts = preds[0]
    for p in preds[1:]:
        pred_ts = pred_ts.concatenate(p)
    return pred_ts

# ---------- NEW: tuning helpers ----------
def _tft_from_trial(trial):
    hp = {
        "hidden_size": trial.suggest_int("hidden_size", 32, 64, step=8),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "num_attention_heads": trial.suggest_categorical("num_attention_heads", [1, 2]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        # 👇 FIX: log scale can't include 0; start at a tiny positive value
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True),
        "n_epochs": MAX_EPOCHS_TUNE,
    }
    return hp

def _val_mae_level_space(model: TFTModel,
                         y_tr: TimeSeries, y_val: TimeSeries,
                         pc_train_s: TimeSeries, ft_all_s: TimeSeries,
                         y_scaler: Scaler) -> float:
    model.fit(series=y_tr,
              past_covariates=pc_train_s.slice_intersect(y_tr),
              future_covariates=ft_all_s.slice(y_tr.start_time(), y_tr.end_time()),
              val_series=y_val,
              val_past_covariates=pc_train_s.slice_intersect(y_val),
              val_future_covariates=ft_all_s.slice(y_val.start_time(), y_val.end_time()),
              verbose=True)
    # Predict exactly len(y_val) starting after y_tr
    yhat_val_s = model.predict(
        n=len(y_val),
        series=y_tr,
        past_covariates=pc_train_s,
        future_covariates=ft_all_s,
        verbose=True,
    )
    yhat_val = y_scaler.inverse_transform(yhat_val_s)
    y_val_unscaled = y_scaler.inverse_transform(y_val)
    err = np.abs(y_val_unscaled.values() - yhat_val.values()).mean()
    return float(err)

def run_quick_tuning(y_tr, y_val, pc_train_s, ft_all_s, y_scaler):
    if not OPTUNA_AVAILABLE:
        print("⚠️ Optuna not available; skipping tuning.")
        return None
    print(f"🔎 Quick tuning with Optuna: trials={N_TRIALS}, epochs/trial={MAX_EPOCHS_TUNE}")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    def objective(trial):
        hp = _tft_from_trial(trial)
        model = make_model_first_stage(hp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mae = _val_mae_level_space(model, y_tr, y_val, pc_train_s, ft_all_s, y_scaler)
        return mae
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    print(f"✅ Best trial: {study.best_trial.number}, MAE={study.best_value:.4f}")
    print(f"   Best params: {study.best_trial.params}")
    return study.best_trial.params

# ======================= MAIN =======================
def main():
    set_repro()

    print("📊 Loading data...")
    train_df, test_df = load_data(TRAIN_CSV, TEST_CSV)
    print(f"✅ Data loaded: train={len(train_df)} rows, test={len(test_df)} rows")

    print("🔄 Building future covariates...")
    ft_train, ft_all = build_future_covs(train_df[DT_COL], test_df[DT_COL])
    print("✅ Future covariates ready")

    print(f"🔄 Building lagged past covariates {LAGS}...")
    pc_train, aligned_targets_df = build_target_lag_covs(train_df, TARGETS, LAGS)
    print("✅ Past covariates built and targets aligned")

    y_all = to_multivar_series(aligned_targets_df, TARGETS)

    print("🔄 Scaling series and covariates...")
    scaled = scale_all(y_all, pc_train, ft_train, ft_all)
    y_all_s     = scaled.y_all_s
    y_scaler    = scaled.y_scaler
    pc_train_s  = scaled.pc_train_s
    pc_scaler   = scaled.pc_scaler
    ft_train_s  = scaled.ft_train_s
    ft_all_s    = scaled.ft_all_s
    print("✅ Scaling complete")

    print("🔄 Splitting train/validation...")
    y_tr, y_val = split_train_val(y_all_s, VAL_FRAC)
    print("✅ Split done")

    # ---------- (Optional) quick tuning ----------
    best_hp = None
    if USE_TUNING:
        best_hp = run_quick_tuning(y_tr, y_val, pc_train_s, ft_all_s, y_scaler)

    # ---------- First-stage training (with validation) ----------
    print("🧠 Building TFT (stage 1)…")
    model_stage1 = make_model_first_stage(best_hp)
    print("🧠 Training (stage 1)…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_stage1.fit(
            series=y_tr,
            past_covariates=pc_train_s.slice_intersect(y_tr),
            future_covariates=ft_train_s.slice_intersect(y_tr),
            val_series=y_val,
            val_past_covariates=pc_train_s.slice_intersect(y_val),
            val_future_covariates=ft_train_s.slice_intersect(y_val),
            verbose=True,
        )

    # ---------- Second-stage fine-tuning on full aligned history ----------
    print("🧠 Fine-tuning on full aligned history (stage 2)…")
    finetune_hp = dict(best_hp or {})
    finetune_hp["n_epochs"] = MAX_EPOCHS
    model_stage2 = make_model_finetune(finetune_hp)
    try:
        model_stage2.load_weights(model_stage1)
    except Exception:
        try:
            model_stage2.model.load_state_dict(model_stage1.model.state_dict())
        except Exception:
            pass
    model_stage2.fit(
        series=y_all_s,
        past_covariates=pc_train_s,
        future_covariates=ft_train_s,
        verbose=True,
    )

    # ---------- Rolling forecast with lag updates ----------
    delta_caps = compute_delta_caps(aligned_targets_df, TARGETS, q=DELTA_CAP_Q)
    print("🧯 Delta caps (abs/hour):", delta_caps)

    print(f"🔮 Forecasting {FORECAST_STEPS} steps with rolling lag updates…")
    work_df_unscaled = aligned_targets_df.copy()  # [id] + targets
    yhat_test = rolling_predict_with_lags(
        model=model_stage2,
        y_hist_s=y_all_s,
        ft_all_s=ft_all_s,
        work_df_unscaled=work_df_unscaled,
        y_scaler=y_scaler,
        pc_scaler=pc_scaler,
        targets=TARGETS,
        lags=LAGS,
        steps=FORECAST_STEPS,
        delta_caps=delta_caps,
        delta_shrink=DELTA_SHRINK,
        ema_alpha=EMA_SMOOTH_ALPHA,
    )

    # ---------- Build submission ----------
    print("🧾 Building submission…")
    pred_df = yhat_test.to_dataframe()  # index = timestamps, cols = TARGETS
    sub = test_df[[DT_COL]].merge(pred_df, left_on=DT_COL, right_index=True, how="left")
    if CLIP_NEGATIVES:
        for col in TARGETS:
            sub[col] = sub[col].astype(float).clip(lower=0.0)
    sub[DT_COL] = pd.to_datetime(sub[DT_COL]).dt.strftime(TIMESTAMP_FMT)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"💾 Wrote submission → {OUT_PATH}")
    print("✅ Done.")

if __name__ == "__main__":
    main()
