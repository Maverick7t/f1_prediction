"""Retrain models from Supabase pipeline tables.

This is intentionally simple:
- Build a training dataframe by joining `qualifying_raw` + `results_raw`
- Compute engineered features
- Train XGBoost binary classifiers for winner and podium
- Save artifacts to `config.MODEL_DIR`
- Register runs in MLflow

It is designed to run as a batch job after races.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Ensure backend/ is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import xgboost as xgb

from utils.config import config, ensure_directories
from services.mlflow_manager import initialize_mlflow, register_model


def _sb():
    from supabase import create_client

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)


def _load_training_df() -> pd.DataFrame:
    sb = _sb()

    # Pull only the columns we need (avoid huge json payloads)
    qresp = (
        sb.table("qualifying_raw")
        .select("race_key,race_year,event,circuit,driver,team,qualifying_position")
        .execute()
    )
    rresp = (
        sb.table("results_raw")
        .select("race_key,race_year,event,circuit,driver,team,finishing_position")
        .execute()
    )

    qdf = pd.DataFrame(qresp.data or [])
    rdf = pd.DataFrame(rresp.data or [])

    if qdf.empty or rdf.empty:
        raise RuntimeError("qualifying_raw/results_raw empty; nothing to train on")

    df = qdf.merge(
        rdf[["race_key", "driver", "finishing_position"]],
        on=["race_key", "driver"],
        how="inner",
    )

    # Basic hygiene
    df["race_year"] = pd.to_numeric(df.get("race_year"), errors="coerce")
    df["qualifying_position"] = pd.to_numeric(df.get("qualifying_position"), errors="coerce")
    df["finishing_position"] = pd.to_numeric(df.get("finishing_position"), errors="coerce")

    df = df.dropna(subset=["race_year", "qualifying_position", "finishing_position", "driver"]).copy()

    # API feature computation expects these fields to exist
    df["event_date"] = pd.NaT

    return df.reset_index(drop=True)


def _compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Reuse the same feature code the API uses
    from app.api import compute_basic_features

    # Ensure required columns exist
    if "team" not in df.columns:
        df["team"] = "Unknown"
    if "circuit" not in df.columns:
        df["circuit"] = df.get("event", "Unknown")

    out = compute_basic_features(df.copy())
    return out


def _prepare_xy(df: pd.DataFrame):
    df = df.copy()

    df["is_winner"] = (df["finishing_position"] == 1).astype(int)
    df["is_podium"] = (df["finishing_position"] <= 3).astype(int)

    num_cols = [
        "qualifying_position",
        "TeamPerfScore",
        "EloRating",
        "RecentFormAvg",
        "CircuitHistoryAvg",
        "DriverExperienceScore",
    ]

    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    encoders: Dict[str, LabelEncoder] = {}
    for c in ["driver", "team", "circuit"]:
        le = LabelEncoder()
        df[c] = df[c].fillna("Unknown").astype(str)
        df[f"{c}_enc"] = le.fit_transform(df[c])
        encoders[c] = le

    feature_cols = num_cols + ["driver_enc", "team_enc", "circuit_enc"]

    X = df[feature_cols].copy()
    y_win = df["is_winner"].copy()
    y_pod = df["is_podium"].copy()

    return X, y_win, y_pod, encoders, feature_cols


def _train_xgb(X: pd.DataFrame, y: pd.Series, *, random_state: int = 42) -> xgb.XGBClassifier:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    clf = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=spw,
    )
    clf.fit(X, y)
    return clf


def retrain_from_supabase(*, training_data_version: str | None = None) -> None:
    ensure_directories()

    print("Loading training data from Supabase...")
    df = _load_training_df()
    print(f"Rows joined: {len(df)}")

    print("Computing engineered features...")
    df = _compute_engineered_features(df)

    X, y_win, y_pod, encoders, feature_cols = _prepare_xy(df)

    print("Training winner model...")
    model_win = _train_xgb(X, y_win)

    print("Training podium model...")
    model_pod = _train_xgb(X, y_pod)

    # Save artifacts
    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    win_path = model_dir / config.MODEL_WINNER_FILE
    pod_path = model_dir / config.MODEL_PODIUM_FILE
    meta_path = model_dir / config.MODEL_METADATA_FILE

    joblib.dump({"model": model_win}, win_path)
    joblib.dump({"model": model_pod}, pod_path)
    joblib.dump({"encoders": encoders, "feature_cols": feature_cols}, meta_path)

    print(f"Saved: {win_path.name}, {pod_path.name}, {meta_path.name}")

    # Register to MLflow (lightweight metrics on train set)
    try:
        initialize_mlflow()
        train_acc_win = float(((model_win.predict_proba(X)[:, 1] >= 0.5).astype(int) == y_win.values).mean())
        train_acc_pod = float(((model_pod.predict_proba(X)[:, 1] >= 0.5).astype(int) == y_pod.values).mean())

        ver = training_data_version or f"supabase_{datetime.now(timezone.utc).date().isoformat()}"

        register_model(
            model_name="xgb_winner",
            model_version="auto",
            model_path=str(win_path),
            metrics={"train_accuracy": train_acc_win},
            features=feature_cols,
            training_data_version=ver,
        )
        register_model(
            model_name="xgb_podium",
            model_version="auto",
            model_path=str(pod_path),
            metrics={"train_accuracy": train_acc_pod},
            features=feature_cols,
            training_data_version=ver,
        )
        print("Registered models in MLflow")
    except Exception as e:
        print(f"MLflow registration skipped: {e}")


if __name__ == "__main__":
    retrain_from_supabase()
