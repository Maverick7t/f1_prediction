#!/usr/bin/env python
"""
Predict all F1 races for the current season and store in database
Fetches qualifying data from cache and generates predictions for all races
Dynamically fetches the race schedule from FastF1 instead of hardcoding.
"""
import os
import sys
import json
import traceback
import pandas as pd
import fastf1
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

# Enable FastF1 cache
fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))


def get_race_schedule(year=None):
    """Dynamically fetch race schedule from FastF1"""
    if year is None:
        year = datetime.now().year
    
    schedule = fastf1.get_event_schedule(year)
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    
    races = []
    for _, event in schedule.iterrows():
        round_num = int(event.get('RoundNumber', 0))
        if round_num == 0:  # Skip testing/pre-season
            continue
        races.append({
            "round": round_num,
            "name": event.get('EventName', 'Unknown'),
            "circuit": event.get('Location', '') or event.get('Country', ''),
            "date": event.get('EventDate').strftime('%Y-%m-%d') if pd.notna(event.get('EventDate')) else ''
        })
    
    return races, year

def predict_all_races():
    """Predict all races for current season using default qualifying (top drivers)"""
    
    # Get dynamic schedule
    races, year = get_race_schedule()
    total_races = len(races)
    
    logger.info("=" * 80)
    logger.info(f"PREDICTING ALL {year} F1 RACES ({total_races} races)")
    logger.info("=" * 80)
    
    predictions_added = 0
    
    # Default qualifying order (based on current season strength)
    default_qualifying = [
        {"driver": "VER", "team": "Red Bull"},
        {"driver": "NOR", "team": "McLaren"},
        {"driver": "LEC", "team": "Ferrari"},
        {"driver": "HAM", "team": "Ferrari"},
        {"driver": "RUS", "team": "Mercedes"},
        {"driver": "SAI", "team": "Williams"},
        {"driver": "PIA", "team": "McLaren"},
        {"driver": "ALO", "team": "Aston Martin"},
        {"driver": "HUL", "team": "Sauber"},
        {"driver": "TSU", "team": "Red Bull"},
    ]
    
    for race in races:
        round_num = race["round"]
        race_name = race["name"]
        circuit = race["circuit"]
        date_str = race["date"]
        
        try:
            logger.info(f"\n[{round_num}/21] {race_name} ({date_str})")
            
            # Convert to DataFrame
            qual_df = pd.DataFrame(default_qualifying)
            
            # Generate race key
            race_key = f"{year}_{round_num}_{circuit.replace(' ', '_')}"
            
            # Run predictions
            predictions = infer_from_qualifying(
                qual_df,
                race_key,
                year,
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
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 80)
    logger.info(f"PREDICTION COMPLETE: {predictions_added}/{total_races} races predicted")
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
