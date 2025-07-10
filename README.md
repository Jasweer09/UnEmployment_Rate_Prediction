**📈 Unemployment Rate Prediction using ML & FastAPI**
A production-ready machine learning system to predict U.S. Unemployment Rate based on key economic indicators. Built using Ridge Regression, tracked and managed with MLflow, and deployed with a FastAPI backend and a responsive frontend.

🚀 **Project Overview**
This project aims to forecast the U.S. unemployment rate using historical economic data such as job openings, civilian labor participation rate, insurance claims, and employment figures. The model uses lag and rolling statistical features to capture temporal patterns and trends over time.

🔧 **Key Features**
✅ Feature Engineering: Lag, rolling mean, and rolling standard deviation features.

✅ Modeling: Ridge Regression (best performing), Decision Tree, Random Forest, XGBoost, and LightGBM evaluated.

✅ Model Tracking: MLflow used to track experiments, parameters, and metrics.

✅ API Service: FastAPI for serving the model with real-time prediction capability.

✅ Frontend: A professional web UI for users to input data and view predictions.

✅ Future-Aware Handling: If a user provides a future date, the system extrapolates features carefully with a disclaimer.

✅ Caching: History cache to store prediction data and maintain context for rolling/lags.

✅ Scaling: Uses same StandardScaler used during model training for consistent preprocessing.

📊 **Input Features**
The system expects monthly economic data as input:

Feature	Description	Typical Range
JTSJOL	Job openings (in thousands)	6000 - 12000
CIVPART	Civilian labor force participation rate (%)	60 - 64
ICSA	Initial claims for state unemployment insurance (weekly)	200,000 - 400,000
ICNSA	Initial claims (non-seasonally adjusted)	180,000 - 380,000
CCSA	Continued claims (seasonally adjusted)	1.5M - 2.5M
CCNSA	Continued claims (non-seasonally adjusted)	1.2M - 2.4M
PAYEMS	Total nonfarm payroll employment (in thousands)	130,000 - 155,000

🔁 The system automatically calculates:

Lag features (last 4 months)

Rolling mean and standard deviation (3 & 6 months)

Date parts (month, year, quarter)

📦 **Tech Stack**
Layer	Technology
📘 Model	Ridge Regression (MLflow model registry)
🧠 ML Ops	MLflow, joblib, StandardScaler
⚙️ Backend	FastAPI
🌐 Frontend	HTML5, Bootstrap, JS (optional deployment on Render/Vercel)
📁 Storage	CSV (for history & caching)
🐍 Language	Python 3.10+

📂 **Folder Structure**
bash
Copy
Edit
project/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── model_loader.py          # MLflow model loader
│   ├── feature_engineering.py   # Lag/Rolling generation
│   ├── history_cache.csv        # Input & prediction cache
│   └── frontend/                # HTML UI
├── mlflow.db                    # MLflow tracking database
├── requirements.txt             # Python dependencies
├── .dockerignore / Dockerfile   # Containerization files
└── README.md
🧪 How to Use
1. 🔨 **Train & Log Model**
Run your notebook/script to train Ridge Regression

Log model to MLflow with name: FinalRidgeModel

2. 🚀 **Start the FastAPI Server**
bash
Copy
Edit
uvicorn app.main:app --reload
3. 🌐** Access Frontend**
Open app/frontend/index.html in browser or deploy it via Render / Vercel. Connects directly to FastAPI.

🔍 **Sample API Request**
json
Copy
Edit
POST /predict/

{
  "date": "2025-01",
  "JTSJOL": 11000,
  "CIVPART": 62.1,
  "ICSA": 250000,
  "ICNSA": 240000,
  "CCSA": 1950000,
  "CCNSA": 1900000,
  "PAYEMS": 160000
}
✅ **Response:**

json
Copy
Edit
{
  "predicted_UNRATE": 4.93,
  "note": "This is a future prediction based on past trends and may not be fully accurate."
}
