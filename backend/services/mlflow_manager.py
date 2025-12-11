"""
MLflow Model Registry & Tracking Manager
Handles model versioning, metrics logging, and model registry
"""

import mlflow
import mlflow.xgboost
from datetime import datetime
import json
from pathlib import Path
import logging
import os

# Load configuration
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "f1-predictions-prod")
REGISTRY_FILE = Path("models/model_registry.json")

# Ensure model registry file exists
REGISTRY_FILE.parent.mkdir(exist_ok=True)

def initialize_mlflow():
    """Initialize MLflow tracking server (local file-based)"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"✓ MLflow initialized: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.error(f"Error initializing MLflow: {e}")
        raise


def register_model(
    model_name: str,
    model_version: str,
    model_path: str,
    metrics: dict,
    features: list,
    training_data_version: str = "2018-2024"
):
    """
    Register a model in MLflow with metrics and metadata
    
    Args:
        model_name: Name of model (e.g., 'xgb_winner')
        model_version: Version tag (e.g., 'v3')
        model_path: Path to joblib model file
        metrics: Dict with {accuracy, f1, precision, recall, etc}
        features: List of feature names used
        training_data_version: Training data version
    
    Returns:
        run_id: MLflow run ID
    """
    try:
        with mlflow.start_run(run_name=f"{model_name}-{model_version}"):
            # Log parameters
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("version", model_version)
            mlflow.log_param("training_data", training_data_version)
            mlflow.log_param("num_features", len(features))
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, float(metric_value))
            
            # Log features as artifact
            features_artifact = {"features": features}
            with open("temp_features.json", "w") as f:
                json.dump(features_artifact, f)
            mlflow.log_artifact("temp_features.json", artifact_path="features")
            Path("temp_features.json").unlink()  # Clean up
            
            # Set tags
            mlflow.set_tag("model_type", "xgboost")
            mlflow.set_tag("framework", "scikit-learn")
            mlflow.set_tag("registered_date", datetime.now().isoformat())
            mlflow.set_tag("status", "production")
            mlflow.set_tag("data_version", training_data_version)
            
            # Log model source
            mlflow.log_text(f"Model stored at: {model_path}", "model_info.txt")
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"✓ Registered {model_name} {model_version} (MLflow run: {run_id})")
            
            return run_id
            
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise


def log_prediction(
    race_name: str,
    predicted_winner: str,
    actual_winner: str = None,
    confidence: float = None,
    model_version: str = "v3"
):
    """
    Log a prediction with outcome (for accuracy tracking)
    
    Args:
        race_name: Name of race
        predicted_winner: Driver code of prediction
        actual_winner: Actual winner (if known)
        confidence: Confidence percentage
        model_version: Model version used
    """
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'race': race_name,
            'predicted': predicted_winner,
            'actual': actual_winner,
            'correct': predicted_winner == actual_winner if actual_winner else None,
            'confidence': confidence,
            'model_version': model_version
        }
        
        # Append to CSV log
        import pandas as pd
        log_path = Path("monitoring/predictions.csv")
        log_path.parent.mkdir(exist_ok=True)
        
        df = pd.DataFrame([log_entry])
        if log_path.exists():
            df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(log_path, mode='w', header=True, index=False)
            
        logger.debug(f"✓ Logged prediction: {predicted_winner} → {race_name}")
        
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")


def get_model_metrics():
    """
    Get latest model metrics from MLflow runs
    
    Returns:
        dict with {winner_model, podium_model, timestamp}
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            return {"error": "No experiments found"}
        
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=10)
        
        metrics_data = {
            "total_runs": len(runs),
            "latest_run": None,
            "models": []
        }
        
        for run in runs:
            model_data = {
                "run_id": run.info.run_id,
                "model_name": run.data.params.get("model_name", "unknown"),
                "version": run.data.params.get("version", "unknown"),
                "status": run.info.status,
                "metrics": run.data.metrics,
                "timestamp": datetime.fromtimestamp(run.info.start_time / 1000).isoformat()
            }
            metrics_data["models"].append(model_data)
            
            if metrics_data["latest_run"] is None:
                metrics_data["latest_run"] = model_data
        
        logger.info(f"✓ Retrieved metrics for {len(runs)} runs")
        return metrics_data
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return {"error": str(e)}


def get_prediction_accuracy():
    """
    Calculate prediction accuracy from logged predictions
    
    Returns:
        dict with {total, correct, accuracy, recent_accuracy}
    """
    try:
        import pandas as pd
        log_path = Path("monitoring/predictions.csv")
        
        if not log_path.exists():
            return {
                "total": 0,
                "correct": 0,
                "accuracy": 0,
                "recent_accuracy": 0,
                "trend": []
            }
        
        df = pd.read_csv(log_path)
        df['correct'] = df['correct'].astype(bool)
        
        total = len(df)
        correct = df['correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Recent accuracy (last 10 predictions)
        recent_total = min(10, len(df))
        recent_correct = df.tail(recent_total)['correct'].sum()
        recent_accuracy = (recent_correct / recent_total * 100) if recent_total > 0 else 0
        
        trend = df.tail(20)['correct'].astype(int).tolist()
        
        logger.info(f"✓ Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        return {
            "total_predictions": total,
            "correct_predictions": int(correct),
            "overall_accuracy": round(accuracy, 2),
            "recent_accuracy": round(recent_accuracy, 2),
            "recent_count": recent_total,
            "trend": trend
        }
        
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return {"error": str(e)}


def get_model_registry():
    """
    Get model registry from MLflow (simplified view for frontend)
    
    Returns:
        dict with current production models and their metrics
    """
    try:
        metrics = get_model_metrics()
        accuracy = get_prediction_accuracy()
        
        registry = {
            "timestamp": datetime.now().isoformat(),
            "mlflow_runs": metrics.get("total_runs", 0),
            "models": [],
            "prediction_stats": accuracy,
            "mlflow_ui_url": "http://localhost:5000/mlflow"
        }
        
        # Extract current production models
        if metrics.get("models"):
            for model_data in metrics["models"]:
                if model_data["version"] in ["v3", "v2"]:  # Current versions
                    registry["models"].append({
                        "name": model_data["model_name"],
                        "version": model_data["version"],
                        "status": "production",
                        "metrics": model_data["metrics"],
                        "timestamp": model_data["timestamp"],
                        "run_id": model_data["run_id"]
                    })
        
        logger.debug(f"✓ Model registry compiled: {len(registry['models'])} models")
        return registry
        
    except Exception as e:
        logger.error(f"Error compiling registry: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


# Initialize on import
try:
    initialize_mlflow()
except Exception as e:
    logger.warning(f"MLflow initialization warning: {e}")
