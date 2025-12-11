"""
F1 Prediction API - Configuration Management
Loads settings from environment variables with sensible defaults
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# =============================================================================
# BASE PATHS
# =============================================================================
# Get the directory where this config file lives (utils/)
# Go up to backend/ parent directory
BASE_DIR = Path(__file__).resolve().parent.parent


def get_path(env_var: str, default: str) -> Path:
    """Get path from env var, resolving relative paths from BASE_DIR"""
    value = os.getenv(env_var, default)
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # ==========================================================================
    # DATA PATHS
    # ==========================================================================
    DATA_PATH = get_path("DATA_PATH", "./f1_training_dataset_2018_2024.csv")
    
    # Model directory and files
    MODEL_DIR = get_path("MODEL_DIR", "./models_spencer")
    MODEL_WINNER_FILE = os.getenv("MODEL_WINNER_FILE", "xgb_winner.joblib")
    MODEL_PODIUM_FILE = os.getenv("MODEL_PODIUM_FILE", "xgb_podium.joblib")
    MODEL_METADATA_FILE = os.getenv("MODEL_METADATA_FILE", "metadata.joblib")
    
    # Full paths to model files
    @property
    def META_FILE(self) -> Path:
        return self.MODEL_DIR / self.MODEL_METADATA_FILE
    
    @property
    def MODEL_WIN_FILE(self) -> Path:
        return self.MODEL_DIR / self.MODEL_WINNER_FILE
    
    @property
    def MODEL_POD_FILE(self) -> Path:
        return self.MODEL_DIR / self.MODEL_PODIUM_FILE
    
    # Cache directories
    CACHE_DIR = get_path("CACHE_DIR", "./cache")
    FASTF1_CACHE_DIR = get_path("FASTF1_CACHE_DIR", "./f1_cache")
    
    @property
    def IMG_CACHE_DIR(self) -> Path:
        return self.CACHE_DIR / "images"
    
    # ==========================================================================
    # DATABASE (Supabase)
    # ==========================================================================
    DATABASE_URL = os.getenv("DATABASE_URL")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    
    @property
    def USE_DATABASE(self) -> bool:
        """Check if database is configured"""
        return bool(self.DATABASE_URL or self.SUPABASE_URL)
    
    @property
    def USE_SUPABASE(self) -> bool:
        """Check if Supabase is configured"""
        return bool(self.SUPABASE_URL and self.SUPABASE_KEY)
    
    # ==========================================================================
    # REDIS CACHE (Optional - for production)
    # ==========================================================================
    REDIS_URL = os.getenv("REDIS_URL")
    
    @property
    def USE_REDIS(self) -> bool:
        """Check if Redis is configured"""
        return bool(self.REDIS_URL)
    
    # ==========================================================================
    # EXTERNAL APIs
    # ==========================================================================
    ERGAST_BASE_URL = os.getenv("ERGAST_BASE_URL", "http://ergast.com/api/f1")
    OPENF1_API_URL = os.getenv("OPENF1_API_URL", "https://api.openf1.org/v1")
    
    # ==========================================================================
    # MLFLOW TRACKING
    # ==========================================================================
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "f1-predictions-prod")
    
    # ==========================================================================
    # CONFIDENCE THRESHOLDS
    # ==========================================================================
    CONFIDENCE_THRESHOLDS = {
        "very_high": 80,
        "high": 60,
        "moderate": 40,
        "low": 0
    }
    
    CONFIDENCE_COLORS = {
        "very_high": "#10b981",  # Green
        "high": "#f59e0b",       # Orange
        "moderate": "#f43f5e",   # Pink
        "low": "#ef4444"         # Red
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = "development"


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = "production"
    
    def __init__(self):
        super().__init__()
        # Validate required production settings
        if not self.SECRET_KEY or self.SECRET_KEY == "dev-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be set in production!")


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# =============================================================================
# CONFIG FACTORY
# =============================================================================
def get_config():
    """Get configuration based on FLASK_ENV"""
    env = os.getenv("FLASK_ENV", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    config_class = configs.get(env, DevelopmentConfig)
    return config_class()


# Create a global config instance
config = get_config()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        config.CACHE_DIR,
        config.IMG_CACHE_DIR,
        config.FASTF1_CACHE_DIR,
        config.MODEL_DIR,
        Path("monitoring"),
        Path("mlruns")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration (for debugging)"""
    print("\n" + "="*60)
    print("F1 PREDICTION API - CONFIGURATION")
    print("="*60)
    print(f"Environment:      {config.FLASK_ENV}")
    print(f"Debug Mode:       {config.DEBUG}")
    print(f"Host:             {config.HOST}:{config.PORT}")
    print(f"Data Path:        {config.DATA_PATH}")
    print(f"Model Dir:        {config.MODEL_DIR}")
    print(f"Cache Dir:        {config.CACHE_DIR}")
    print(f"FastF1 Cache:     {config.FASTF1_CACHE_DIR}")
    print(f"Use Database:     {config.USE_DATABASE}")
    print(f"Use Supabase:     {config.USE_SUPABASE}")
    print(f"Use Redis:        {config.USE_REDIS}")
    print(f"MLflow URI:       {config.MLFLOW_TRACKING_URI}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test configuration loading
    print_config()
    ensure_directories()
    print("✓ All directories created successfully")
