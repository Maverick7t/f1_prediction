#!/usr/bin/env python
"""Check if Supabase database has necessary data"""
import os
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from config import config
from database_v2 import get_qualifying_cache, get_prediction_logger

print("=" * 80)
print("DATABASE DATA AVAILABILITY CHECK")
print("=" * 80)

# Check 1: Qualifying Cache in Supabase
print("\n[1] QUALIFYING DATA IN SUPABASE")
print("-" * 80)
try:
    cache = get_qualifying_cache(config)
    
    # Get latest cached qualifying
    latest = cache.get_latest_cached_qualifying()
    if latest:
        if isinstance(latest, str):
            latest = json.loads(latest)
        
        if isinstance(latest, list):
            print(f"[OK] Latest qualifying cached: {len(latest)} drivers")
            if latest:
                print(f"  First driver: {latest[0].get('driver', latest[0].get('code', 'N/A'))}")
                print(f"  Keys: {list(latest[0].keys())}")
        elif isinstance(latest, dict):
            print(f"[OK] Latest qualifying cached (dict format)")
            print(f"  Keys: {list(latest.keys())}")
    else:
        print("[ERROR] No qualifying data cached in Supabase")
        
except Exception as e:
    print(f"[ERROR] Error checking qualifying cache: {e}")

# Check 2: Predictions in Supabase
print("\n[2] PREDICTIONS DATA IN SUPABASE")
print("-" * 80)
try:
    logger = get_prediction_logger(config)
    
    # Get prediction history
    all_preds = logger.get_prediction_history(limit=1000)
    if len(all_preds) > 0:
        print(f"[OK] Predictions table has data: {len(all_preds)} rows")
        print(f"  Latest prediction: {all_preds.iloc[0]['created_at']}")
        print(f"  Columns: {list(all_preds.columns)}")
    else:
        print("[ERROR] No predictions in Supabase")
        
except Exception as e:
    print(f"[ERROR] Error checking predictions: {e}")

# Check 3: Local Training Data
print("\n[3] LOCAL TRAINING DATA")
print("-" * 80)
try:
    HIST_CSV = str(config.DATA_PATH)
    parquet_path = Path(HIST_CSV).with_suffix('.parquet')
    
    import pandas as pd
    
    if parquet_path.exists():
        hist_data = pd.read_parquet(parquet_path)
        print(f"[OK] Historical data (parquet): {len(hist_data)} rows")
        print(f"  Columns: {len(hist_data.columns)}")
        print(f"  Year range: {hist_data.get('race_year', pd.Series()).min()} - {hist_data.get('race_year', pd.Series()).max()}")
        print(f"  Drivers: {hist_data['driver'].nunique()}")
        print(f"  Circuits: {hist_data['circuit'].nunique()}")
    elif Path(HIST_CSV).exists():
        hist_data = pd.read_csv(HIST_CSV)
        print(f"[OK] Historical data (CSV): {len(hist_data)} rows")
    else:
        print(f"[ERROR] Training data file not found: {HIST_CSV}")
        
except Exception as e:
    print(f"[ERROR] Error checking training data: {e}")

# Check 4: Models
print("\n[4] ML MODELS")
print("-" * 80)
try:
    import joblib
    
    META_FILE = str(config.META_FILE)
    MODEL_WIN_FILE = str(config.MODEL_WIN_FILE)
    MODEL_POD_FILE = str(config.MODEL_POD_FILE)
    
    meta = joblib.load(META_FILE)
    model_win = joblib.load(MODEL_WIN_FILE)
    model_pod = joblib.load(MODEL_POD_FILE)
    
    print(f"[OK] Metadata loaded: {len(meta['feature_cols'])} features")
    print(f"[OK] Win model loaded: {type(model_win['model']).__name__}")
    print(f"[OK] Podium model loaded: {type(model_pod['model']).__name__}")
    
except Exception as e:
    print(f"[ERROR] Error checking models: {e}")

# Check 5: Feature Store Snapshots
print("\n[5] FEATURE STORE SNAPSHOTS")
print("-" * 80)
try:
    from feature_store import get_feature_store
    
    fs = get_feature_store(config)
    health = fs.health_check()
    
    print(f"[OK] Feature store initialized")
    print(f"  Drivers cached: {health['drivers_in_memory']}")
    print(f"  Parquet exists: {health['parquet_exists']}")
    print(f"  Redis connected: {health['redis_connected']}")
    
except Exception as e:
    print(f"[ERROR] Error checking feature store: {e}")

print("\n" + "=" * 80)
print("DATABASE CHECK COMPLETE")
print("=" * 80)
