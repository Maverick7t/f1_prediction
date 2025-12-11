"""
F1 Prediction API - Database Layer
Supports both local CSV mode and production PostgreSQL (Supabase)
"""

import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Try to import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.info("Supabase client not installed. Using CSV mode.")


class DatabaseManager:
    """
    Database abstraction layer that supports:
    - Local mode: CSV files (development)
    - Production mode: Supabase PostgreSQL
    """
    
    def __init__(self, config):
        self.config = config
        self.supabase: Optional[Client] = None
        self._hist_data: Optional[pd.DataFrame] = None
        self._mode = "csv"  # "csv" or "supabase"
        
        self._initialize()
    
    def _initialize(self):
        """Initialize database connection based on config"""
        if self.config.USE_DATABASE and SUPABASE_AVAILABLE:
            try:
                self.supabase = create_client(
                    self.config.SUPABASE_URL,
                    self.config.SUPABASE_KEY
                )
                self._mode = "supabase"
                logger.info("✓ Connected to Supabase database")
            except Exception as e:
                logger.warning(f"Supabase connection failed: {e}. Falling back to CSV.")
                self._mode = "csv"
        else:
            self._mode = "csv"
            logger.info("✓ Using local CSV mode")
    
    @property
    def mode(self) -> str:
        """Return current database mode"""
        return self._mode
    
    # =========================================================================
    # HISTORICAL RACE DATA
    # =========================================================================
    
    def get_historical_data(self) -> pd.DataFrame:
        """Get all historical race data"""
        if self._mode == "supabase":
            return self._get_historical_from_supabase()
        else:
            return self._get_historical_from_csv()
    
    def _get_historical_from_csv(self) -> pd.DataFrame:
        """Load historical data from CSV file"""
        if self._hist_data is None:
            csv_path = self.config.DATA_PATH
            if csv_path.exists():
                self._hist_data = pd.read_csv(csv_path)
                logger.info(f"✓ Loaded {len(self._hist_data)} rows from CSV")
            else:
                logger.error(f"CSV file not found: {csv_path}")
                self._hist_data = pd.DataFrame()
        return self._hist_data.copy()
    
    def _get_historical_from_supabase(self) -> pd.DataFrame:
        """Load historical data from Supabase"""
        try:
            response = self.supabase.table('races').select('*').execute()
            df = pd.DataFrame(response.data)
            logger.info(f"✓ Loaded {len(df)} rows from Supabase")
            return df
        except Exception as e:
            logger.error(f"Supabase query failed: {e}. Falling back to CSV.")
            return self._get_historical_from_csv()
    
    def get_races_by_year(self, year: int) -> pd.DataFrame:
        """Get races for a specific year"""
        if self._mode == "supabase":
            try:
                response = self.supabase.table('races').select('*').eq('race_year', year).execute()
                return pd.DataFrame(response.data)
            except Exception as e:
                logger.error(f"Query failed: {e}")
                
        # Fallback to CSV filter
        df = self.get_historical_data()
        return df[df['race_year'] == year]
    
    def get_driver_history(self, driver_code: str, limit: int = 20) -> pd.DataFrame:
        """Get recent races for a specific driver"""
        if self._mode == "supabase":
            try:
                response = (self.supabase.table('races')
                           .select('*')
                           .eq('driver', driver_code)
                           .order('event_date', desc=True)
                           .limit(limit)
                           .execute())
                return pd.DataFrame(response.data)
            except Exception as e:
                logger.error(f"Query failed: {e}")
        
        # Fallback to CSV filter
        df = self.get_historical_data()
        driver_df = df[df['driver'] == driver_code].sort_values('event_date', ascending=False)
        return driver_df.head(limit)
    
    def get_qualifying_data(self, race_key: str = None, race_year: int = None, 
                           circuit: str = None, event: str = None) -> pd.DataFrame:
        """Get qualifying data for a race"""
        df = self.get_historical_data()
        
        if race_key:
            result = df[(df.get('race_key') == race_key) & df['qualifying_position'].notna()]
            if not result.empty:
                return result.sort_values('qualifying_position')
        
        if race_year and circuit:
            result = df[(df['race_year'] == int(race_year)) & 
                       (df.get('circuit', '').astype(str).str.lower() == str(circuit).lower()) & 
                       df['qualifying_position'].notna()]
            if not result.empty:
                return result.sort_values('qualifying_position')
        
        if race_year and event:
            result = df[(df['race_year'] == int(race_year)) & 
                       (df.get('event', '').astype(str).str.lower().str.contains(str(event).lower())) & 
                       df['qualifying_position'].notna()]
            if not result.empty:
                return result.sort_values('qualifying_position')
        
        return pd.DataFrame()
    
    # =========================================================================
    # PREDICTIONS LOGGING
    # =========================================================================
    
    def log_prediction(self, race_name: str, predicted_winner: str, 
                      actual_winner: str = None, confidence: float = None,
                      model_version: str = "v3") -> bool:
        """Log a prediction for tracking accuracy"""
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'race': race_name,
            'predicted': predicted_winner,
            'actual': actual_winner,
            'correct': predicted_winner == actual_winner if actual_winner else None,
            'confidence': confidence,
            'model_version': model_version
        }
        
        if self._mode == "supabase":
            try:
                self.supabase.table('predictions').insert(prediction).execute()
                logger.debug(f"✓ Logged prediction to Supabase: {predicted_winner} → {race_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to log prediction to Supabase: {e}")
        
        # Fallback to CSV logging
        return self._log_prediction_to_csv(prediction)
    
    def _log_prediction_to_csv(self, prediction: dict) -> bool:
        """Log prediction to local CSV file"""
        try:
            log_path = Path("monitoring/predictions.csv")
            log_path.parent.mkdir(exist_ok=True)
            
            df = pd.DataFrame([prediction])
            if log_path.exists():
                df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(log_path, mode='w', header=True, index=False)
            
            logger.debug(f"✓ Logged prediction to CSV: {prediction['predicted']} → {prediction['race']}")
            return True
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            return False
    
    def get_prediction_history(self, limit: int = 100) -> pd.DataFrame:
        """Get prediction history for accuracy tracking"""
        if self._mode == "supabase":
            try:
                response = (self.supabase.table('predictions')
                           .select('*')
                           .order('timestamp', desc=True)
                           .limit(limit)
                           .execute())
                return pd.DataFrame(response.data)
            except Exception as e:
                logger.error(f"Query failed: {e}")
        
        # Fallback to CSV
        log_path = Path("monitoring/predictions.csv")
        if log_path.exists():
            df = pd.read_csv(log_path)
            return df.tail(limit)
        return pd.DataFrame()
    
    def get_prediction_accuracy(self) -> dict:
        """Calculate prediction accuracy statistics"""
        df = self.get_prediction_history(limit=1000)
        
        if df.empty:
            return {
                "total_predictions": 0,
                "correct_predictions": 0,
                "overall_accuracy": 0,
                "recent_accuracy": 0,
                "recent_count": 0,
                "trend": []
            }
        
        # Convert correct column to boolean
        df['correct'] = df['correct'].astype(bool)
        
        total = len(df)
        correct = df['correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Recent accuracy (last 10)
        recent_df = df.tail(10)
        recent_total = len(recent_df)
        recent_correct = recent_df['correct'].sum()
        recent_accuracy = (recent_correct / recent_total * 100) if recent_total > 0 else 0
        
        # Trend (last 20 predictions)
        trend = df.tail(20)['correct'].astype(int).tolist()
        
        return {
            "total_predictions": total,
            "correct_predictions": int(correct),
            "overall_accuracy": round(accuracy, 2),
            "recent_accuracy": round(recent_accuracy, 2),
            "recent_count": recent_total,
            "trend": trend
        }
    
    # =========================================================================
    # DRIVER & TEAM DATA
    # =========================================================================
    
    def get_all_drivers(self) -> List[str]:
        """Get list of all driver codes"""
        df = self.get_historical_data()
        return df['driver'].unique().tolist()
    
    def get_all_teams(self) -> List[str]:
        """Get list of all team names"""
        df = self.get_historical_data()
        return df['team'].unique().tolist()
    
    def get_driver_stats(self, driver_code: str) -> dict:
        """Get statistics for a driver"""
        df = self.get_historical_data()
        driver_df = df[df['driver'] == driver_code]
        
        if driver_df.empty:
            return {}
        
        return {
            "driver": driver_code,
            "total_races": len(driver_df),
            "wins": len(driver_df[driver_df['finishing_position'] == 1]),
            "podiums": len(driver_df[driver_df['finishing_position'] <= 3]),
            "avg_finish": driver_df['finishing_position'].mean(),
            "teams": driver_df['team'].unique().tolist(),
            "seasons": sorted(driver_df['race_year'].unique().tolist())
        }
    
    # =========================================================================
    # HEALTH CHECK
    # =========================================================================
    
    def health_check(self) -> dict:
        """Check database connectivity and status"""
        status = {
            "mode": self._mode,
            "connected": False,
            "record_count": 0,
            "last_race_year": None
        }
        
        try:
            df = self.get_historical_data()
            status["connected"] = True
            status["record_count"] = len(df)
            status["last_race_year"] = int(df['race_year'].max()) if not df.empty else None
        except Exception as e:
            status["error"] = str(e)
        
        return status


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_db_instance: Optional[DatabaseManager] = None

def get_database(config=None) -> DatabaseManager:
    """Get or create database manager singleton"""
    global _db_instance
    
    if _db_instance is None:
        if config is None:
            from config import config as app_config
            config = app_config
        _db_instance = DatabaseManager(config)
    
    return _db_instance


def reset_database():
    """Reset database instance (for testing)"""
    global _db_instance
    _db_instance = None
