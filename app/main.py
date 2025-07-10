from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.featue_engineering import prepare_features
from app.model_loader import load_ridge_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_ridge_model()

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the Pydantic model for input validation
class UserInput(BaseModel):
    date: str
    JTSJOL: float
    CIVPART: float
    ICSA: float
    ICNSA: float
    CCSA: float
    CCNSA: float
    PAYEMS: float

# Function to ensure history cache exists
def initialize_cache():
    if not os.path.exists("app/history_cache.csv"):
        empty_df = pd.DataFrame(columns=[
            'date', 'JTSJOL', 'CIVPART', 'ICSA', 'ICNSA', 'CCSA', 'CCNSA', 'PAYEMS', 'UNRATE'
        ])
        empty_df.to_csv("app/history_cache.csv", index=False)
        print("❗ Initialized empty history_cache.csv.")
    else:
        # Check if the file is empty
        if os.stat("app/history_cache.csv").st_size == 0:
            empty_df = pd.DataFrame(columns=[
                'date', 'JTSJOL', 'CIVPART', 'ICSA', 'ICNSA', 'CCSA', 'CCNSA', 'PAYEMS', 'UNRATE'
            ])
            empty_df.to_csv("app/history_cache.csv", index=False)
            print("❗ history_cache.csv was empty. Initialized with columns.")

# Ensure cache file exists
initialize_cache()

@app.post("/predict/")
async def predict(input_data: UserInput):
    try:
        initialize_cache()
        print("📥 Received user input: ", input_data.dict())

        # Convert input to a dataframe
        user_data = pd.DataFrame([input_data.dict()])

        # Load historical data (in this case, 'history_cache.csv')
        history_data = pd.read_csv("app/history_cache.csv")
        scaler = StandardScaler()
        
        print(f"📊 Loaded history data. Current data size: {history_data.shape}")

        # Append the new data point (user's input)
        new_data = pd.concat([history_data, user_data], ignore_index=True)
        
        print(f"📝 Appended new data. Data size after append: {new_data.shape}")

        # Prepare features (generate lag and rolling features)
        prepared_data = prepare_features(new_data, [
            'JTSJOL', 'CIVPART', 'ICSA', 'ICNSA', 'CCSA', 'CCNSA', 'PAYEMS', 'UNRATE'])

        print(f"⚙️ Prepared features. Columns after feature engineering: {prepared_data.columns}")
        
        # Extract last row (latest data after features were added)
        data = prepared_data.drop(columns = ['date'], axis = 1)
        scaler.fit_transform(data.iloc[:-1])
        last_row = data.iloc[-1:]
        print('type: ',type(last_row))
        # Drop the target column to prevent leakage
        print(f"🔍 Features for prediction: {last_row.shape}")
        
        
        
        print(f"🔍 Features for prediction: {last_row.values}", last_row.values.size)
        # Predict UNRATE
        
        predicted_unrate = model.predict(scaler.transform(last_row))
        
        print(f"🔮 Prediction result: {predicted_unrate[0]}")

        # Save this prediction to the history cache (so it's used for future predictions)
        #last_row['UNRATE'] = np.abs(predicted_unrate[0])
        last_row.to_csv("app/history_cache.csv", mode='a', header=False, index=False)

        print(f"📈 Updated history cache with new prediction.")

        # Return the prediction with a note about future predictions
        return {
            "predicted_UNRATE": np.abs(predicted_unrate[0]),
            "note": "This is a future prediction based on past trends and may not be fully accurate."
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")