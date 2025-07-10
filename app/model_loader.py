import mlflow
from mlflow.sklearn import load_model

def load_ridge_model(model_uri: str = "model/artifacts"):
    """
    Load Ridge regression model from MLflow registry.
    """
    try:
        mlflow.set_tracking_uri("./mlruns")
        model = load_model(model_uri)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print("❌ Error loading model:", str(e))
        return None