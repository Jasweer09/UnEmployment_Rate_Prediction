from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from app.featue_engineering import prepare_features
from app.model_loader import load_ridge_model
import os

model = load_ridge_model()

app = FastAPI()

class UserInput(BaseModel):
    date: str
    JTSJOL: float
    CIVPART: float
    ICSA: float
    ICNSA: float
    CCSA: float
    CCNSA: float
    PAYEMS: float


@app.post("/predict/")
async def predict(input_data: UserInput):
    try:
        # Convert input to a dataframe
        user_data = pd.DataFrame([input_data.dict()])
        
        # Load historical data (in this case, 'history_cache.csv')
        history_data = pd.read_csv("app/history_cache.csv")
        
        # Append the new data point (user's input)
        new_data = pd.concat([history_data, user_data], ignore_index=True)

        # Prepare features (generate lag and rolling features)
        prepared_data = prepare_features(new_data, ['JTSJOL', 'CIVPART', 'ICSA', 'ICNSA', 'CCSA', 'CCNSA', 'PAYEMS', 'UNRATE'])

        # Extract last row (latest data after features were added)
        last_row = prepared_data.iloc[-1:]

        # Drop the target column to prevent leakage
        X = last_row.drop(columns=['UNRATE'])

        # Predict UNRATE
        predicted_unrate = model.predict(X)

        # Save this prediction to the history cache (so it's used for future predictions)
        last_row['UNRATE'] = predicted_unrate[0]
        last_row.to_csv("app/history_cache.csv", mode='a', header=False, index=False)

        # Return the prediction with a note about future predictions
        return {
            "predicted_UNRATE": predicted_unrate[0],
            "note": "This is a future prediction based on past trends and may not be fully accurate."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")
