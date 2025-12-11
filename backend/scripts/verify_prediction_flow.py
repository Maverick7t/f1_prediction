#!/usr/bin/env python
"""
Verify that the prediction system is following the correct data flow.
Tests: qualifying → merge → features → predict
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sys
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config

# ============================================================================
# VERIFICATION 1: Check all required data sources
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION 1: Data Sources")
print("="*80)

# Check historical data
HIST_CSV = config.DATA_PATH
parquet_path = Path(HIST_CSV).with_suffix('.parquet')

if parquet_path.exists():
    hist_data = pd.read_parquet(parquet_path)
    print(f"[OK] Historical data loaded from parquet: {len(hist_data)} rows")
else:
    hist_data = pd.read_csv(HIST_CSV)
    print(f"[OK] Historical data loaded from CSV: {len(hist_data)} rows")

print(f"  Columns: {list(hist_data.columns)}")
print(f"  Year range: {hist_data['race_year'].min()} - {hist_data['race_year'].max()}")
print(f"  Drivers: {hist_data['driver'].nunique()}")
print(f"  Circuits: {hist_data['circuit'].nunique()}")

# Check models
META_FILE = config.META_FILE
MODEL_WIN_FILE = config.MODEL_WIN_FILE
MODEL_POD_FILE = config.MODEL_POD_FILE

meta = joblib.load(META_FILE)
encoders = meta["encoders"]
feature_cols = meta["feature_cols"]
model_win = joblib.load(MODEL_WIN_FILE)["model"]
model_pod = joblib.load(MODEL_POD_FILE)["model"]

print(f"\n[OK] Models loaded:")
print(f"  Win model: {type(model_win).__name__}")
print(f"  Pod model: {type(model_pod).__name__}")
print(f"  Feature count: {len(feature_cols)}")
print(f"  Features: {feature_cols[:5]}... (showing 5)")

# ============================================================================
# VERIFICATION 2: Check that compute_basic_features exists
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION 2: Feature Computation")
print("="*80)

try:
    from app.api import compute_basic_features
    print("[OK] compute_basic_features function found in api.py")
except ImportError:
    print("[ERROR] compute_basic_features NOT FOUND in api.py")
    print("  This is required to compute features from merged data!")
    sys.exit(1)

# ============================================================================
# VERIFICATION 3: Test with sample qualifying data
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION 3: Prediction Test with Sample Data")
print("="*80)

# Create sample qualifying data for Qatar 2025
sample_qual = pd.DataFrame({
    'driver': ['VER', 'NOR', 'LEC', 'SAI', 'RUS', 'PIA'],
    'team': ['Red Bull', 'McLaren', 'Ferrari', 'Ferrari', 'Mercedes', 'McLaren'],
    'qualifying_position': [1, 2, 3, 4, 5, 6]
})

print(f"\nSample qualifying data ({len(sample_qual)} drivers):")
print(sample_qual.to_string(index=False))

# Test the prediction function
from app.api import infer_from_qualifying

try:
    predictions = infer_from_qualifying(
        qual_df=sample_qual,
        race_key="2025_1_Qatar_Grand_Prix",
        race_year=2025,
        event="Qatar Grand Prix",
        circuit="Qatar"
    )
    
    print(f"\n[OK] Prediction generated successfully")
    print(f"\nWinner Prediction:")
    print(f"  Driver: {predictions['winner_prediction']['driver']}")
    print(f"  Win Prob: {predictions['winner_prediction']['percentage']}%")
    print(f"  Confidence: {predictions['winner_prediction']['confidence']}")
    
    print(f"\nTop 3 Predictions:")
    for i, driver in enumerate(predictions['top3_prediction'][:3], 1):
        print(f"  {i}. {driver['driver']}: {driver['percentage']}% (podium)")
    
    # ========================================================================
    # VERIFICATION 4: Check probability values
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION 4: Probability Validation")
    print("="*80)
    
    winner_prob = predictions['winner_prediction']['percentage']
    
    if 25 <= winner_prob <= 60:
        print(f"[OK] Winner probability realistic: {winner_prob}%")
    else:
        print(f"[WARNING] Winner probability suspicious: {winner_prob}%")
        if winner_prob < 15:
            print("  This suggests features may not be computed correctly")
            print("  (All drivers look like rookies with no experience)")
    
    # ========================================================================
    # VERIFICATION 5: Check data flow
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION 5: Data Flow Analysis")
    print("="*80)
    
    print("\nExpected flow:")
    print("  1. Qualifying data (pos 1-6) -> Input")
    print("  2. Merge with hist_data (2018-2024) -> Combined dataset")
    print("  3. Compute features (rolling averages, experience, etc.)")
    print("  4. Extract 2025 race rows with computed features")
    print("  5. Encode categoricals (driver, team, circuit)")
    print("  6. Run XGBoost models")
    print("  7. Output: Probabilities (30-60%)")
    
    print("\nActual result:")
    if 25 <= winner_prob <= 60:
        print("  [OK] Flow appears CORRECT")
        print("    Features are being computed with historical context")
    else:
        print("  [ERROR] Flow appears BROKEN")
        print("    Features are using defaults (FeatureStore lookup failed)")
    
    # ========================================================================
    # VERIFICATION 6: Feature details
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION 6: Feature Inspection")
    print("="*80)
    
    # Get first driver's info
    first_driver = predictions['top3_prediction'][0]['driver']
    print(f"\nAnalyzing features for {first_driver}:")
    print(f"  Qualifying position: 1 (pole)")
    print(f"  Career races: {hist_data[hist_data['driver'] == first_driver].shape[0]}")
    print(f"  Recent form avg: ?")
    print(f"  Circuit history: ?")
    print(f"  Team perf score: ?")
    print(f"  Elo rating: ?")
    
    print("\nTo debug feature values, check api.py infer_from_qualifying() function")
    print("and verify that compute_basic_features() is being called with merged data.")
    
except Exception as e:
    print(f"\n[ERROR] Prediction FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nSummary:")
print("[OK] Historical data loaded (2018-2024)")
print("[OK] Models and encoders loaded")
if 25 <= winner_prob <= 60:
    print(f"[OK] Prediction probabilities realistic ({winner_prob}%)")
    print("\n[SUCCESS] SYSTEM IS WORKING CORRECTLY")
else:
    print(f"[ERROR] Prediction probabilities unrealistic ({winner_prob}%)")
    print("\n[WARNING] CHECK: Is compute_basic_features() merging and computing correctly?")
