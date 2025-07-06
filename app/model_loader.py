import mlflow
from mlflow.sklearn import load_model

def load_ridge_model(model_uri: str = "models:/Final Ridge Model/Production"):
    """
    Load Ridge regression model from MLflow registry.
    """
    try:
        model = load_model(model_uri)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print("❌ Error loading model:", str(e))
        return None
