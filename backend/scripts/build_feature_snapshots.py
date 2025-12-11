#!/usr/bin/env python3
"""
F1 Prediction System - Build Feature Snapshots

Run this script to pre-compute driver and circuit features from historical data.
This generates small snapshot files used at inference time instead of loading
the full 60MB CSV on every prediction.

Usage:
    python build_feature_snapshots.py

Output:
    - models_spencer/driver_features_snapshot.parquet (~50KB)
    - models_spencer/circuit_features_snapshot.parquet (~100KB)
    - data/historical_races.parquet (compressed version of CSV)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config


def load_historical_data() -> pd.DataFrame:
    """Load historical CSV data"""
    csv_path = config.DATA_PATH
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    logger.info(f"Loading historical data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def convert_to_parquet(df: pd.DataFrame, output_path: Path):
    """Convert DataFrame to Parquet format (columnar, compressed)"""
    logger.info(f"Converting to Parquet: {output_path}")
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to Parquet with snappy compression
    df.to_parquet(output_path, compression='snappy', index=False)
    
    # Compare sizes
    csv_size = config.DATA_PATH.stat().st_size / (1024 * 1024)  # MB
    parquet_size = output_path.stat().st_size / (1024 * 1024)  # MB
    
    logger.info(f"✓ CSV size: {csv_size:.2f} MB → Parquet size: {parquet_size:.2f} MB")
    logger.info(f"  Compression ratio: {csv_size/parquet_size:.1f}x smaller")


def compute_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute driver features from historical data.
    
    For each driver, compute:
    - RecentFormAvg: Rolling 5-race average finishing position
    - EloRating: Latest ELO rating
    - TotalRaces: Total races driven
    - DriverExperienceScore: Normalized experience (0-1)
    - AvgFinishPosition: Career average finish
    - Wins: Total wins
    - Podiums: Total podiums
    - LastTeam: Most recent team
    - LastRaceDate: Most recent race
    """
    logger.info("Computing driver features...")
    
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.sort_values(["driver", "event_date"])
    
    # Convert finishing position to numeric
    df["finishing_position_num"] = pd.to_numeric(df["finishing_position"], errors="coerce")
    
    # Compute rolling features
    df["RecentFormAvg"] = df.groupby("driver")["finishing_position_num"] \
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Get latest values per driver
    latest = df.groupby("driver").last().reset_index()
    
    # Aggregate stats per driver
    driver_stats = df.groupby("driver").agg({
        "finishing_position_num": ["mean", "count"],
        "points": "sum",
        "event_date": "max",
        "team": "last"
    }).reset_index()
    
    # Flatten column names
    driver_stats.columns = ["driver", "AvgFinishPosition", "TotalRaces", 
                           "TotalPoints", "LastRaceDate", "LastTeam"]
    
    # Add wins and podiums
    wins = df[df["finishing_position_num"] == 1].groupby("driver").size().reset_index(name="Wins")
    podiums = df[df["finishing_position_num"] <= 3].groupby("driver").size().reset_index(name="Podiums")
    
    driver_stats = driver_stats.merge(wins, on="driver", how="left")
    driver_stats = driver_stats.merge(podiums, on="driver", how="left")
    driver_stats["Wins"] = driver_stats["Wins"].fillna(0).astype(int)
    driver_stats["Podiums"] = driver_stats["Podiums"].fillna(0).astype(int)
    
    # Add latest RecentFormAvg and EloRating
    latest_features = latest[["driver", "RecentFormAvg"]].copy()
    if "EloRating" in latest.columns:
        latest_features["EloRating"] = latest["EloRating"]
    else:
        latest_features["EloRating"] = 1500.0
    
    driver_stats = driver_stats.merge(latest_features, on="driver", how="left")
    
    # Compute experience score (normalized 0-1)
    max_races = driver_stats["TotalRaces"].max()
    driver_stats["DriverExperienceScore"] = (driver_stats["TotalRaces"] / max_races).clip(0, 1)
    
    # Fill NaN values
    driver_stats["RecentFormAvg"] = driver_stats["RecentFormAvg"].fillna(10.0)
    driver_stats["EloRating"] = driver_stats["EloRating"].fillna(1500.0)
    driver_stats["AvgFinishPosition"] = driver_stats["AvgFinishPosition"].fillna(10.0)
    
    # Convert LastRaceDate to string for Parquet
    driver_stats["LastRaceDate"] = driver_stats["LastRaceDate"].astype(str)
    
    logger.info(f"✓ Computed features for {len(driver_stats)} drivers")
    
    return driver_stats


def compute_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute driver performance per circuit.
    
    For each (driver, circuit) pair:
    - CircuitAvgFinish: Average finishing position at this circuit
    - CircuitRaces: Number of races at this circuit
    - CircuitBestFinish: Best finish at this circuit
    """
    logger.info("Computing circuit features...")
    
    df = df.copy()
    df["finishing_position_num"] = pd.to_numeric(df["finishing_position"], errors="coerce")
    
    circuit_stats = df.groupby(["driver", "circuit"]).agg({
        "finishing_position_num": ["mean", "count", "min"]
    }).reset_index()
    
    # Flatten column names
    circuit_stats.columns = ["driver", "circuit", "CircuitAvgFinish", 
                            "CircuitRaces", "CircuitBestFinish"]
    
    # Fill NaN
    circuit_stats["CircuitAvgFinish"] = circuit_stats["CircuitAvgFinish"].fillna(10.0)
    
    logger.info(f"✓ Computed circuit features: {len(circuit_stats)} (driver, circuit) pairs")
    
    return circuit_stats


def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute team performance per season.
    
    For each (team, year):
    - TeamAvgFinish: Average finishing position
    - TeamPoints: Total points
    - TeamPerfScore: Normalized performance (0-1, higher is better)
    """
    logger.info("Computing team features...")
    
    df = df.copy()
    df["finishing_position_num"] = pd.to_numeric(df["finishing_position"], errors="coerce")
    
    team_stats = df.groupby(["race_year", "team"]).agg({
        "finishing_position_num": "mean",
        "points": "sum"
    }).reset_index()
    
    team_stats.columns = ["race_year", "team", "TeamAvgFinish", "TeamPoints"]
    
    # Compute normalized performance score per season
    def normalize_team_perf(series):
        max_val = series.max()
        min_val = series.min()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        # Lower avg finish = better, so invert
        return (max_val - series) / (max_val - min_val)
    
    team_stats["TeamPerfScore"] = team_stats.groupby("race_year")["TeamAvgFinish"] \
        .transform(normalize_team_perf)
    
    logger.info(f"✓ Computed team features: {len(team_stats)} (team, year) pairs")
    
    return team_stats


def main():
    """Build all feature snapshots"""
    print("=" * 60)
    print("F1 PREDICTION SYSTEM - BUILD FEATURE SNAPSHOTS")
    print("=" * 60)
    print()
    
    # Load data
    df = load_historical_data()
    
    # Output directory
    output_dir = config.MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("-" * 60)
    
    # 1. Convert CSV to Parquet
    parquet_path = config.DATA_PATH.with_suffix('.parquet')
    convert_to_parquet(df, parquet_path)
    
    print()
    print("-" * 60)
    
    # 2. Build driver features snapshot
    driver_features = compute_driver_features(df)
    driver_output = output_dir / "driver_features_snapshot.parquet"
    driver_features.to_parquet(driver_output, compression='snappy', index=False)
    
    driver_size = driver_output.stat().st_size / 1024  # KB
    logger.info(f"✓ Saved driver features: {driver_output} ({driver_size:.1f} KB)")
    
    # Show sample
    print("\nSample driver features:")
    print(driver_features[["driver", "RecentFormAvg", "EloRating", "TotalRaces", "Wins"]].head(10).to_string())
    
    print()
    print("-" * 60)
    
    # 3. Build circuit features snapshot
    circuit_features = compute_circuit_features(df)
    circuit_output = output_dir / "circuit_features_snapshot.parquet"
    circuit_features.to_parquet(circuit_output, compression='snappy', index=False)
    
    circuit_size = circuit_output.stat().st_size / 1024  # KB
    logger.info(f"✓ Saved circuit features: {circuit_output} ({circuit_size:.1f} KB)")
    
    print()
    print("-" * 60)
    
    # 4. Build team features snapshot
    team_features = compute_team_features(df)
    team_output = output_dir / "team_features_snapshot.parquet"
    team_features.to_parquet(team_output, compression='snappy', index=False)
    
    team_size = team_output.stat().st_size / 1024  # KB
    logger.info(f"✓ Saved team features: {team_output} ({team_size:.1f} KB)")
    
    print()
    print("=" * 60)
    print("BUILD COMPLETE!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  • {parquet_path} (historical data, columnar)")
    print(f"  • {driver_output} ({driver_size:.1f} KB)")
    print(f"  • {circuit_output} ({circuit_size:.1f} KB)")
    print(f"  • {team_output} ({team_size:.1f} KB)")
    print()
    print("These snapshots are used at inference time instead of")
    print("loading the full CSV on every prediction request.")
    print()
    
    # Summary stats
    total_kb = driver_size + circuit_size + team_size
    csv_mb = config.DATA_PATH.stat().st_size / (1024 * 1024)
    print(f"Total snapshot size: {total_kb:.1f} KB")
    print(f"Original CSV size:   {csv_mb:.2f} MB")
    print(f"Inference data reduction: {csv_mb * 1024 / total_kb:.0f}x smaller")


if __name__ == "__main__":
    main()
