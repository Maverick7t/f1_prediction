#!/usr/bin/env python
"""
Predict all 2025 F1 races and store in database
Fetches qualifying data from cache and generates predictions for all races
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from utils.config import config
from database.database_v2 import get_prediction_logger, get_qualifying_cache
from app.api import infer_from_qualifying, get_races_with_predictions_and_history
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database connections
prediction_logger = get_prediction_logger(config)
qualifying_cache = get_qualifying_cache(config)

# Hardcoded 2025 race schedule with proper names
RACES_2025 = [
    {"round": 1, "name": "Bahrain Grand Prix", "circuit": "Bahrain", "date": "2025-03-16"},
    {"round": 2, "name": "Saudi Arabian Grand Prix", "circuit": "Saudi Arabia", "date": "2025-03-30"},
    {"round": 3, "name": "Australian Grand Prix", "circuit": "Australia", "date": "2025-04-13"},
    {"round": 4, "name": "Japanese Grand Prix", "circuit": "Japan", "date": "2025-04-27"},
    {"round": 5, "name": "Chinese Grand Prix", "circuit": "China", "date": "2025-05-11"},
    {"round": 6, "name": "Monaco Grand Prix", "circuit": "Monaco", "date": "2025-05-25"},
    {"round": 7, "name": "Canadian Grand Prix", "circuit": "Canada", "date": "2025-06-08"},
    {"round": 8, "name": "Spanish Grand Prix", "circuit": "Spain", "date": "2025-06-22"},
    {"round": 9, "name": "Austrian Grand Prix", "circuit": "Austria", "date": "2025-07-06"},
    {"round": 10, "name": "British Grand Prix", "circuit": "Britain", "date": "2025-07-20"},
    {"round": 11, "name": "Hungarian Grand Prix", "circuit": "Hungary", "date": "2025-08-03"},
    {"round": 12, "name": "Belgian Grand Prix", "circuit": "Belgium", "date": "2025-08-24"},
    {"round": 13, "name": "Dutch Grand Prix", "circuit": "Netherlands", "date": "2025-09-07"},
    {"round": 14, "name": "Italian Grand Prix", "circuit": "Italy", "date": "2025-09-21"},
    {"round": 15, "name": "Azerbaijan Grand Prix", "circuit": "Azerbaijan", "date": "2025-10-05"},
    {"round": 16, "name": "United States Grand Prix", "circuit": "United States", "date": "2025-10-19"},
    {"round": 17, "name": "Mexico City Grand Prix", "circuit": "Mexico", "date": "2025-10-26"},
    {"round": 18, "name": "São Paulo Grand Prix", "circuit": "Brazil", "date": "2025-11-09"},
    {"round": 19, "name": "Las Vegas Grand Prix", "circuit": "Las Vegas", "date": "2025-11-22"},
    {"round": 20, "name": "Qatar Grand Prix", "circuit": "Qatar", "date": "2025-11-30"},
    {"round": 21, "name": "Abu Dhabi Grand Prix", "circuit": "Abu Dhabi", "date": "2025-12-07"},
]

def predict_all_races():
    """Predict all 2025 races using default qualifying (top drivers)"""
    
    logger.info("=" * 80)
    logger.info("PREDICTING ALL 2025 F1 RACES")
    logger.info("=" * 80)
    
    predictions_added = 0
    
    # Default qualifying order (based on 2024 standings)
    default_qualifying = [
        {"driver": "VER", "team": "Red Bull"},
        {"driver": "NOR", "team": "McLaren"},
        {"driver": "LEC", "team": "Ferrari"},
        {"driver": "HAM", "team": "Mercedes"},
        {"driver": "RUS", "team": "Mercedes"},
        {"driver": "SAI", "team": "Ferrari"},
        {"driver": "PIA", "team": "McLaren"},
        {"driver": "ALO", "team": "Aston Martin"},
        {"driver": "HUL", "team": "Haas"},
        {"driver": "TSU", "team": "Racing Bulls"},
    ]
    
    for race in RACES_2025:
        round_num = race["round"]
        race_name = race["name"]
        circuit = race["circuit"]
        date_str = race["date"]
        
        try:
            logger.info(f"\n[{round_num}/21] {race_name} ({date_str})")
            
            # Convert to DataFrame
            qual_df = pd.DataFrame(default_qualifying)
            
            # Generate race key
            race_key = f"2025_{round_num}_{circuit.replace(' ', '_')}"
            
            # Run predictions
            predictions = infer_from_qualifying(
                qual_df,
                race_key,
                2025,
                race_name,
                circuit
            )
            
            predicted_winner = predictions["winner_prediction"]["driver"]
            confidence = predictions["winner_prediction"]["percentage"]
            
            logger.info(f"  Prediction: {predicted_winner} ({confidence}%)")
            
            # Log to database
            success = prediction_logger.log_prediction(
                race_name=race_name,
                predicted_winner=predicted_winner,
                confidence=confidence,
                model_version="xgb_v3"
            )
            
            if success:
                logger.info(f"  ✓ Logged to database")
                predictions_added += 1
        
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 80)
    logger.info(f"PREDICTION COMPLETE: {predictions_added}/21 races predicted")
    logger.info("=" * 80)

def fetch_all_predictions():
    """Fetch all stored predictions from database"""
    
    logger.info("\n" + "=" * 80)
    logger.info("FETCHING ALL 2025 PREDICTIONS FROM DATABASE")
    logger.info("=" * 80)
    
    try:
        history = prediction_logger.get_prediction_history(limit=1000)
        
        if history is None or len(history) == 0:
            logger.warning("No predictions found in database")
            return
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(history)
        
        # Filter only 2025 races
        if 'race' in df.columns:
            df_2025 = df[df['race'].str.contains('Grand Prix', case=False, na=False)]
        else:
            df_2025 = df
        
        logger.info(f"\nTotal predictions: {len(df_2025)}")
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTIONS vs ACTUALS")
        logger.info("=" * 80)
        
        # Display predictions
        display_columns = ['race', 'predicted', 'actual', 'correct', 'confidence', 'timestamp']
        existing_cols = [col for col in display_columns if col in df_2025.columns]
        
        for idx, row in df_2025.iterrows():
            race = row.get('race', 'Unknown')
            predicted = row.get('predicted', 'N/A')
            actual = row.get('actual', 'TBD')
            correct = row.get('correct', None)
            confidence = row.get('confidence', 0)
            
            status = '✓' if correct == True else ('✗' if correct == False else '?')
            
            logger.info(f"\n{race}")
            logger.info(f"  Predicted: {predicted} ({confidence}%)")
            logger.info(f"  Actual: {actual}")
            logger.info(f"  Result: {status}")
        
        # Calculate accuracy
        correct_count = (df_2025['correct'] == True).sum()
        total_count = len(df_2025)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info(f"ACCURACY: {correct_count}/{total_count} ({accuracy:.1f}%)")
        logger.info("=" * 80)
        
        # Export to CSV
        csv_path = "predictions_2025.csv"
        df_2025.to_csv(csv_path, index=False)
        logger.info(f"\nExported to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Step 1: Predict all races
    predict_all_races()
    
    # Step 2: Fetch and display all predictions
    fetch_all_predictions()
