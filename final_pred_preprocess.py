#!/usr/bin/env python3
# final_pred_lags_2.py
# Train a lag-enabled TFT and predict next 504 hours with rolling lag updates.
# Uses feature_engineering.preprocess_dataset() to build future covariates
# (calendar + weather lag 8760), while pollutant lags are still built here.

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

# << NEW: pull in your FE pipeline (no pollutant lags; only weather year-lag) >>
import sys
sys.path.append('src/utils')
from feature_engineering import preprocess_dataset

# make float32 the default everywhere
torch.set_default_dtype(torch.float32)

# ======================= CONFIG =======================
DT_COL = "id"
TARGETS = ["valeur_CO", "valeur_NO2", "valeur_O3", "valeur_PM10", "valeur_PM25"]

# pollutant lags (in hours) -> built HERE as past_covariates
LAGS = (1, 24, 168)   # 1h, 1d, 1w

# weather lags (hours) -> built in FE and passed as FUTURE covariates
WEATHER_LAGS = [8760]  # 365 days * 24 hours

# context / horizon
INPUT_CHUNK_LEN = 336     # ~2 weeks of history
OUTPUT_CHUNK_LEN = 24     # 1 day
FORECAST_STEPS = 504      # required prediction length

# training
VAL_FRAC = 0.10
MAX_EPOCHS = 1            # increase later for better results
BATCH_SIZE = 64
SEED = 42
USE_GPU = torch.cuda.is_available()

# I/O
TRAIN_CSV = "data/train_imputed_all_columns.csv"
TEST_CSV  = "data/test.csv"
OUT_PATH  = "submissions/submission_tft_lags_3.csv"

# safety / post-process
CLIP_NEGATIVES = True
TIMESTAMP_FMT = "%Y-%m-%d %H"

# ----- Smoothness controls (penalize hour-to-hour jumps) -----
DELTA_CAP_Q      = 0.92  # cap abs(delta) to 95th percentile of train deltas
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

# ---------- FUTURE COVARIATES via your FE (calendar + weather lag 8760) ----------
def build_future_covs_with_preprocess(train_df: pd.DataFrame,
                                      test_df: pd.DataFrame) -> Tuple[TimeSeries, TimeSeries]:
    """
    Build future covariates by calling `preprocess_dataset` on a FULL timeline
    [train_start .. test_end + (decoder-1)h], adding weather_lag8760 and calendar/lockdown/curfew.
    No pollutant lags are added here.
    """
    t_train_min = pd.to_datetime(train_df[DT_COL].min()).floor("h")
    t_train_max = pd.to_datetime(train_df[DT_COL].max()).ceil("h")
    t_test_max  = pd.to_datetime(test_df[DT_COL].max()).ceil("h")

    # ensure decoder sees enough future
    full_end = t_test_max + pd.Timedelta(hours=OUTPUT_CHUNK_LEN - 1)
    full_times  = pd.date_range(start=t_train_min, end=full_end, freq="h")
    train_times = pd.date_range(start=t_train_min, end=t_train_max, freq="h")

    # build tiny frames with just id to feed FE
    full_df  = pd.DataFrame({DT_COL: full_times})
    train_id = pd.DataFrame({DT_COL: train_times})

    # try to reuse cached weather if present; else let FE fetch
    df_weather = None
    wcsv = "./data/paris_weather.csv"
    if os.path.exists(wcsv):
        try:
            df_weather = pd.read_csv(wcsv)
        except Exception:
            df_weather = None

    # FE call: weather_lags only; DO NOT add pollutant lags (lags=None)
    full_proc = preprocess_dataset(full_df.copy(), df_weather=df_weather,
                                   weather_lags=WEATHER_LAGS, lags=None)  # :contentReference[oaicite:3]{index=3}
    # Align index name and slice the training view
    if full_proc.index.name != DT_COL:
        full_proc.index.name = DT_COL
    train_proc = full_proc.loc[train_times[0]:train_times[-1]]

    # Use all engineered columns as future covariates
    feat_cols = list(full_proc.columns)

    ft_train = TimeSeries.from_dataframe(
        train_proc.reset_index(), time_col=DT_COL, value_cols=feat_cols,
        fill_missing_dates=True, freq="h"
    ).astype(np.float32)

    ft_all = TimeSeries.from_dataframe(
        full_proc.reset_index(), time_col=DT_COL, value_cols=feat_cols,
        fill_missing_dates=True, freq="h"
    ).astype(np.float32)

    return ft_train, ft_all

# ---------- PAST COVARIATES (pollutant lags) – built here ----------
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

def make_model_first_stage() -> TFTModel:
    return TFTModel(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_size=40,
        lstm_layers=3,
        dropout=0.2,
        num_attention_heads=1,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_EPOCHS,
        add_relative_index=True,
        random_state=SEED,
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        pl_trainer_kwargs=dict(
            accelerator="gpu" if USE_GPU else "cpu",
            devices=1,
            precision="32-true",
            enable_progress_bar=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=False),
            ],
            max_epochs=MAX_EPOCHS,
            log_every_n_steps=50,
        ),
    )

def make_model_finetune() -> TFTModel:
    return TFTModel(
        input_chunk_length=INPUT_CHUNK_LEN,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_size=40,
        lstm_layers=3,
        dropout=0.2,
        num_attention_heads=1,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_EPOCHS,
        add_relative_index=True,
        random_state=SEED,
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        pl_trainer_kwargs=dict(
            accelerator="gpu" if USE_GPU else "cpu",
            devices=1,
            precision="32-true",
            enable_progress_bar=True,
            callbacks=[
                EarlyStopping(monitor="train_loss", mode="min", patience=3, verbose=False),
            ],
            max_epochs=MAX_EPOCHS,
            log_every_n_steps=50,
        ),
    )

# -------- Build rolling PC window from buffer (unchanged) --------
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
    delta_caps: dict,
    delta_shrink: float = DELTA_SHRINK,
    ema_alpha: float = EMA_SMOOTH_ALPHA,
) -> TimeSeries:
    preds: List[TimeSeries] = []
    buf_idxed = work_df_unscaled.set_index(DT_COL).sort_index()

    for _ in range(steps):
        t_next = buf_idxed.index[-1] + pd.Timedelta(hours=1)
        enc_end = t_next - pd.Timedelta(hours=1)

        pc_window = _build_pc_window_from_buffer(
            buf_idxed=buf_idxed,
            end_time=enc_end,
            targets=targets,
            lags=lags,
            window_len=INPUT_CHUNK_LEN + 1,
        ).astype(np.float32)
        pc_window_s = pc_scaler.transform(pc_window).astype(np.float32)

        window_start = pc_window.start_time()
        dec_end = t_next + pd.Timedelta(hours=OUTPUT_CHUNK_LEN - 1)
        ft_window_s = ft_all_s.slice(window_start, dec_end).astype(np.float32)

        yhat_s = model.predict(
            n=1,
            series=y_hist_s,
            past_covariates=pc_window_s,
            future_covariates=ft_window_s,
            verbose=False,
        )
        yhat_lvl = y_scaler.inverse_transform(yhat_s)

        prev_lvl = buf_idxed.iloc[-1][targets].astype(float).values
        raw_pred = yhat_lvl.values(copy=False)[0]
        delta = delta_shrink * (raw_pred - prev_lvl)

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

# ======================= MAIN =======================
def main():
    set_repro()

    print("📊 Loading data...")
    train_df, test_df = load_data(TRAIN_CSV, TEST_CSV)
    print(f"✅ Data loaded: train={len(train_df)} rows, test={len(test_df)} rows")

    # ---------- Build future covariates via FE (calendar + weather lag 8760) ----------
    print("🔄 Building future covariates (FE + weather_lag8760)…")
    ft_train, ft_all = build_future_covs_with_preprocess(train_df, test_df)
    print("✅ Future covariates ready")

    # ---------- Build past covariates from pollutant lags (here) ----------
    print(f"🔄 Building lagged past covariates {LAGS}…")
    pc_train, aligned_targets_df = build_target_lag_covs(train_df, TARGETS, LAGS)
    print("✅ Past covariates built and targets aligned")

    # ---------- Target series aligned with past covariates ----------
    y_all = to_multivar_series(aligned_targets_df, TARGETS)

    # ---------- Scale everything ----------
    print("🔄 Scaling series and covariates...")
    scaled = scale_all(y_all, pc_train, ft_train, ft_all)
    y_all_s     = scaled.y_all_s
    y_scaler    = scaled.y_scaler
    pc_train_s  = scaled.pc_train_s
    pc_scaler   = scaled.pc_scaler
    ft_train_s  = scaled.ft_train_s
    ft_all_s    = scaled.ft_all_s
    print("✅ Scaling complete")

    # ---------- Train/val split ----------
    print("🔄 Splitting train/validation...")
    y_tr, y_val = split_train_val(y_all_s, VAL_FRAC)
    pc_tr  = pc_train_s.slice_intersect(y_tr)
    pc_val = pc_train_s.slice_intersect(y_val)
    ft_tr  = ft_train_s.slice_intersect(y_tr)
    ft_val = ft_train_s.slice_intersect(y_val)
    print("✅ Split done")

    # ---------- First-stage training (with validation) ----------
    print("🧠 Building TFT (stage 1)…")
    model_stage1 = make_model_first_stage()
    print("🧠 Training (stage 1)…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_stage1.fit(
            series=y_tr,
            past_covariates=pc_tr,
            future_covariates=ft_tr,
            val_series=y_val,
            val_past_covariates=pc_val,
            val_future_covariates=ft_val,
            verbose=True,
        )

    # ---------- Second-stage fine-tuning on full aligned history ----------
    print("🧠 Fine-tuning on full aligned history (stage 2)…")
    model_stage2 = make_model_finetune()
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

    # ---------- Caps from training deltas ----------
    delta_caps = compute_delta_caps(aligned_targets_df, TARGETS, q=DELTA_CAP_Q)
    print("🧯 Delta caps (abs/hour):", delta_caps)

    # ---------- Rolling forecast with lag updates ----------
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
    pred_df = yhat_test.to_dataframe()  # index = timestamps

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
