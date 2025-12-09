"""Verify model data and prediction capability"""

import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import json

print("=" * 70)
print("MODEL FILES CHECK")
print("=" * 70)

models_dir = Path('models_spencer')
if models_dir.exists():
    print(f'✓ Models directory exists: {models_dir}')
    for f in models_dir.glob('*'):
        size_mb = f.stat().st_size / (1024*1024)
        print(f'  - {f.name}: {size_mb:.2f} MB')
else:
    print(f'✗ Models directory NOT found')

# Check training data
print()
print("=" * 70)
print("TRAINING DATA CHECK")
print("=" * 70)

csv_path = Path('f1_training_dataset_2018_2024.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f'✓ Training CSV exists: {csv_path}')
    print(f'  - Rows: {len(df)}')
    print(f'  - Columns: {list(df.columns)[:8]}...')  # First 8 columns
    print(f'  - Years: {sorted(df["race_year"].unique())}')
    print(f'  - Driver count: {df["driver"].nunique()}')
    print(f'  - Circuit count: {df["circuit"].nunique()}')
    print(f'  - Finishing positions: {df["finishing_position"].nunique()}')
else:
    print(f'✗ Training CSV NOT found')

# Check feature store
print()
print("=" * 70)
print("FEATURE STORE CHECK")
print("=" * 70)

try:
    from feature_store import get_feature_store
    from config import config
    
    fs = get_feature_store(config)
    drivers = fs.get_all_drivers()
    print(f'✓ Feature store loaded')
    print(f'  - Drivers: {len(drivers)}')
    print(f'  - Sample drivers: {drivers[:5]}')
except Exception as e:
    print(f'✗ Feature store error: {e}')
    import traceback
    traceback.print_exc()

# Check if models can be loaded
print()
print("=" * 70)
print("MODEL LOADING CHECK")
print("=" * 70)

model_files = ['xgb_winner.joblib', 'xgb_podium.joblib']
for model_name in model_files:
    model_path = models_dir / model_name
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print(f'✓ {model_name}: Loaded successfully')
            if hasattr(model, 'n_features_in_'):
                print(f'  - Input features expected: {model.n_features_in_}')
        except Exception as e:
            print(f'✗ {model_name}: Failed to load - {e}')
    else:
        print(f'✗ {model_name}: File not found')

# Check prediction cache
print()
print("=" * 70)
print("PREDICTION CACHE CHECK")
print("=" * 70)

try:
    from prediction_cache import get_prediction_cache
    cache = get_prediction_cache()
    stats = cache.get_stats()
    print(f'✓ Prediction cache initialized')
    print(f'  - Stats: {json.dumps(stats, indent=2)}')
except Exception as e:
    print(f'✗ Prediction cache error: {e}')
    import traceback
    traceback.print_exc()

# Check Supabase connection
print()
print("=" * 70)
print("SUPABASE CHECK")
print("=" * 70)

try:
    from database_v2 import get_prediction_logger
    logger = get_prediction_logger(config)
    print(f'✓ Prediction logger initialized')
    print(f'  - Mode: {logger.mode}')
except Exception as e:
    print(f'✗ Prediction logger error: {e}')

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
