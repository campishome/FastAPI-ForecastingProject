from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor

app = FastAPI()

# origins = [
#     "http://localhost:4200", 
#     "https://rice-project-nine.vercel.app",     
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ใส่ ["*"] ถ้าจะเปิดทุก origin
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"msg": "API is alive!"}

model_paths = {
    "06": {  # กข 6
        "province": {
            "rf": "assets/models/model_province/randomforest_model_province_06.pkl",
            "sgd": "assets/models/model_province/sgd_model_province_06.pkl",
            "linear": "assets/models/model_province/linearreg_model_province_06.pkl",
            "catb": "assets/models/model_province/catboost_model_province_06.pkl",
            "xgb": "assets/models/model_province/xgboost_model_province_06.pkl",
        },
        "district": {
            "rf": "assets/models/model_district/randomforest_model_district_06.pkl",
            "sgd": "assets/models/model_district/sgd_model_district_06.pkl",
            "linear": "assets/models/model_district/linearreg_model_district_06.pkl",
            "catb": "assets/models/model_district/catboost_model_district_06.pkl",
            "xgb": "assets/models/model_district/xgboost_model_district_06.pkl",
        }
    },
    "105": {  # ขาวดอกมะลิ 105
        "province": {
            "rf": "assets/models/model_province/randomforest_model_province_105.pkl",
            "sgd": "assets/models/model_province/sgd_model_province_105.pkl",
            "linear": "assets/models/model_province/linearreg_model_province_105.pkl",
            "catb": "assets/models/model_province/catboost_model_province_105.pkl",
            "xgb": "assets/models/model_province/xgboost_model_province_105.pkl",
        },
        "district": {
            "rf": "assets/models/model_district/randomforest_model_district_105.pkl",
            "sgd": "assets/models/model_district/sgd_model_district_105.pkl",
            "linear": "assets/models/model_district/linearreg_model_district_105.pkl",
            "catb": "assets/models/model_district/catboost_model_district_105.pkl",
            "xgb": "assets/models/model_district/xgboost_model_district_105.pkl",
        }
    }
}

scalers = {
    "06": {
        "province": {
            "X": "assets/models/model_province/scaler_X_province_06.pkl",
            "y": "assets/models/model_province/scaler_y_province_06.pkl"
        },
        "district": {
            "X": "assets/models/model_district/scaler_X_district_06.pkl",
            "y": "assets/models/model_district/scaler_y_district_06.pkl"
        }
    },
    "105": {
        "province": {
            "X": "assets/models/model_province/scaler_X_province_105.pkl",
            "y": "assets/models/model_province/scaler_y_province_06.pkl"
        },
        "district": {
            "X": "assets/models/model_district/scaler_X_district_105.pkl",
            "y": "assets/models/model_district/scaler_y_district_105.pkl"
        }
    }
}

#Features แยกตาม level
features = {
    "province": [
        'Product/ Yield (kg)', 'Plant-month', 'Cutivated Area (rai)',
        'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg',
        'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst',
        'province_lat', 'province_lon','Year'
    ],
    "district": [
        'Product/ Yield (kg)', 'Plant-month', 'Cutivated Area (rai)',
        'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg',
        'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst',
        'district_lat', 'district_lon','Year'
    ]
}

targets = ['Yield/Harvest-area', 'Yield/Plant-area']


@app.post("/forecast/{level}/{crop}/")
async def forecast(
    level: str,
    crop: str,
    model_type: str = Form(...),
    Province: str = Form(None),
    District: str = Form(None),
    file: UploadFile = File(...)
):
    if level not in ["province", "district"]:
        return {"error": "Invalid level. Choose 'province' or 'district'."}

    if crop not in model_paths:
        return {"error": f"Crop '{crop}' not supported"}

    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are supported"}

    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    # Filter ตามระดับ
    col_name = "Province" if level == "province" else "District"
    area_value = Province if level == "province" else District
    if col_name not in df.columns:
        return {"error": f"Column '{col_name}' not found in Excel"}

    filtered_df = df[df[col_name] == area_value]
    if filtered_df.empty:
        return {"error": f"No data found for {col_name} '{area_value}'"}

    # ✅ ใช้ features ตาม level
    X = filtered_df[features[level]]

    # Load scalers + model
    scaler_X = joblib.load(scalers[crop][level]["X"])
    scaler_y = joblib.load(scalers[crop][level]["y"])
    if model_type not in model_paths[crop][level]:
        return {"error": f"Invalid model_type '{model_type}'"}

    model = joblib.load(model_paths[crop][level][model_type])

    # Predict
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Return forecast
    results = filtered_df.copy()
    results[targets] = y_pred

    return {
        "filename": file.filename,
        col_name: area_value,
        "Crop": crop,
        "Model": model_type,
        "Forecast": results[[col_name] + features[level][-2:] + targets].to_dict(orient="records")
    }

@app.post("/evaluate/{level}/{crop}/")
async def evaluate(
    level: str,
    crop: str,
    model_type: str = Form(...),
    Province: str = Form(None),
    District: str = Form(None),
    file: UploadFile = File(...)
):
    if level not in ["province", "district"]:
        return {"error": "Invalid level. Choose 'province' or 'district'."}

    if crop not in model_paths:
        return {"error": f"Crop '{crop}' not supported"}

    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are supported"}

    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    # Filter
    col_name = "Province" if level == "province" else "District"
    area_value = Province if level == "province" else District
    if col_name not in df.columns:
        return {"error": f"Column '{col_name}' not found in Excel"}

    filtered_df = df[df[col_name] == area_value]
    if filtered_df.empty:
        return {"error": f"No data found for {col_name} '{area_value}'"}

    #ใช้ features ตาม level โดยไม่ rename
    X = filtered_df[features[level]]
    y = filtered_df[targets]

    # Load scalers + model
    scaler_X = joblib.load(scalers[crop][level]["X"])
    scaler_y = joblib.load(scalers[crop][level]["y"])
    if model_type not in model_paths[crop][level]:
        return {"error": f"Invalid model_type '{model_type}'"}

    model = joblib.load(model_paths[crop][level][model_type])

    # Predict
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred, multioutput="raw_values"))
    mse = mean_squared_error(y, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y, y_pred, multioutput="raw_values")

    # --- Feature importance ---
    feature_names = X.columns.tolist()
    importance = np.zeros(len(feature_names))
    try:
        if model_type in ["rf", "xgb", "catb"]:
            all_importances = [est.feature_importances_ for est in model.estimators_]
            importance = np.mean(all_importances, axis=0)
        elif model_type in ["linear", "sgd"]:
            all_coefs = [np.abs(est.coef_) for est in model.estimators_]
            importance = np.mean(all_coefs, axis=0)
    except:
        importance = np.zeros(len(feature_names))

    if importance.sum() > 0:
        importance = importance / importance.sum() * 100

    feature_importance = [
        {"feature": f, "importance": round(float(imp), 2)}
        for f, imp in zip(feature_names, importance)
    ]

    return {
        "filename": file.filename,
        col_name: area_value,
        "Crop": crop,
        "Model": model_type,
        "RMSE": rmse.tolist(),
        "MSE": mse.tolist(),
        "MAE": mae.tolist(),
        "y_true_vs_pred": [
            {
                "Year": year,
                "true": yt,
                "pred": yp
            }
            for year, yt, yp in zip(filtered_df["Year"].tolist(), y.values.tolist(), y_pred.tolist())
        ],
        "feature_importance": feature_importance
    }


# python -m uvicorn main:app --reload
# git add .
# git commit -m "update add CORS middleware"
# git push origin main

#--okay


# ====== Schema ======
class FutureForecastRequest(BaseModel):
    model_type: str
    Province: str | None = None
    District: str | None = None
    features: dict   # key = feature name, value = float

# ====== Feature Mapping ตาม level ======
features_province = [
        'Product/ Yield (kg)', 'Plant-month', 'Cutivated Area (rai)',
        'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg',
        'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst',
        'province_lat', 'province_lon','Year'
    ]

features_district = [
        'Product/ Yield (kg)', 'Plant-month', 'Cutivated Area (rai)',
        'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg',
        'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst',
        'district_lat', 'district_lon','Year'
    ]

# ====== Endpoint พยากรณ์อนาคต ======

#input เป็นไฟล์ได้ด้วย  ---อีกเส้น อีก endpoint
#เวลาเลือก จังหวัด อำเภอ ให้ fig lat lon ไว้เลย 
#input สำหรับพยากรณ์มีได้หลาย record และผลการพยากรณ์มีออกมาได้เยอะ เช่น มีข้อมูลมาหลายๆอำเภอ ต้องออกมาเป็นแต่ละอำเภอในปีปัจจุบัน  "Year":2564
#***sample ข้อมูลไว้ใช้ทดสอบเยอะๆหน่อย มีทั้งผิด และถูก
#หาปัจจัยที่มีผล ต่อ target ว่าอันไหน มีผลสุด ---อีกเส้น อีก endpoint

@app.post("/forecast_future/{level}/{crop}/") 
async def forecast_future(level: str, crop: str, request: FutureForecastRequest):
    if level not in ["province", "district"]:
        return {"error": "Invalid level. Choose 'province' or 'district'."}

    if crop not in model_paths:
        return {"error": f"Crop '{crop}' not supported"}

    if request.model_type not in model_paths[crop][level]:
        return {"error": f"Invalid model_type '{request.model_type}'"}

    # Load scalers + model
    scaler_X = joblib.load(scalers[crop][level]["X"])
    scaler_y = joblib.load(scalers[crop][level]["y"])
    model = joblib.load(model_paths[crop][level][request.model_type])

    # กำหนด features ที่ต้องใช้ตาม level
    if level == "province":
        feature_list = features_province
    else:
        feature_list = features_district

    # เตรียม features
    try:
        X_input = np.array([request.features[f] for f in feature_list]).reshape(1, -1)
    except KeyError as e:
        return {"error": f"Missing feature: {str(e)}"}

    # Scale & Predict
    X_scaled = scaler_X.transform(X_input)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return {
        "Province": request.Province,
        "District": request.District,
        "Crop": crop,
        "Model": request.model_type,
        "Forecast": {
            "Yield/Harvest-area": float(y_pred[0][0]),
            "Yield/Plant-area": float(y_pred[0][1])
        }
    }

#python -m uvicorn main:app --reload  #เวลาทำนายปี 2568 → ต้องใส่ features ของปี 2568 (a,b,c,d ของปี 68) เข้าไป
@app.post("/forecast_file/{level}/{crop}/")
async def forecast_file(level: str, crop: str, file: UploadFile = File(...),model_type: str = Form(...),):
    if level not in ["province", "district"]:
        return {"error": "Invalid level. Choose 'province' or 'district'."}

    if crop not in model_paths:
        return {"error": f"Crop '{crop}' not supported"}

    # โหลดโมเดลและสเกลเลอร์
    scaler_X = joblib.load(scalers[crop][level]["X"])
    scaler_y = joblib.load(scalers[crop][level]["y"])

    if model_type not in model_paths[crop][level]:
        return {"error": f"Invalid model_type '{model_type}'"}

    model = joblib.load(model_paths[crop][level][model_type])

    # กำหนด features
    if level == "province":
        feature_list = features_province
    else:
        feature_list = features_district

    # อ่านไฟล์
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith(".csv") else pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}

    # ตรวจสอบว่ามี features ครบ
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        return {"error": f"Missing features in file: {missing_features}"}

    results = []
    for _, row in df.iterrows():
        try:
            X_input = np.array([row[f] for f in feature_list]).reshape(1, -1)
            X_scaled = scaler_X.transform(X_input)
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            result = {
                "Province": row.get("Province") if "Province" in row else None,
                "District": row.get("District") if "District" in row else None,
                "Crop": crop,
                "Variety/types": row.get("Variety/types") if "Variety/types" in row else None,
                "Model": model_type,
                "Year": row.get("Year") if "Year" in row else None,  # ปีปัจจุบัน "Year": pd.Timestamp.now().year,   # ปีปัจจุบัน
                "Forecast": {
                    "Yield/Harvest-area": float(y_pred[0][0]),
                    "Yield/Plant-area": float(y_pred[0][1])
                }
            }
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "row": row.to_dict()})

    return {"results": results}

# train model test เป็น อนาคตต่อจาก train เสมอ **fig เป็นปีสุดท้ายเอาไว้ test เลยดีกว่า
@app.post("/train_model/{level}")
async def train_model(
    level: str,
    model_type: str = Form(...),
    train_years: str = Form(...),
    test_years: str = Form(...),
    file: UploadFile = File(...),
):
    # --- Validate input ---
    if level not in features:
        return {"error": f"Invalid level: {level}, must be one of {list(features.keys())}"}

    # --- Load dataset ---
    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Failed to read Excel file: {e}"}

    # --- Convert year params ---
    try:
        train_years = [int(y.strip()) for y in train_years.split(",")]
        test_years = [int(y.strip()) for y in test_years.split(",")]
    except ValueError:
        return {"error": "Train/test years must be comma-separated integers."}

    # --- Validate targets ---
    for target in targets:
        if target not in df.columns:
            return {"error": f"Target '{target}' not found in dataset"}

    if "Year" not in df.columns:
        return {"error": "Column 'Year' not found in dataset"}

    # --- Split train/test ---
    train_df = df[df["Year"].isin(train_years)]
    test_df = df[df["Year"].isin(test_years)]

    if train_df.empty or test_df.empty:
        return {"error": "Train or test set is empty after filtering by year."}

    # --- Prepare features ---
    feature_cols = features[level]
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        return {"error": f"Missing features in dataset: {missing_features}"}

    X_train = train_df[feature_cols].drop(columns=["Year"], errors="ignore")
    y_train = train_df[targets]
    X_test = test_df[feature_cols].drop(columns=["Year"], errors="ignore")
    y_test = test_df[targets].values

    # --- Scaling ---
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    # --- Choose model ---
    if model_type == "rf":
        base_estimator = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "sgd":
        base_estimator = SGDRegressor(random_state=42)
    elif model_type == "linear":
        base_estimator = LinearRegression()
    elif model_type == "catb":
        base_estimator = CatBoostRegressor(verbose=0, random_state=42)
    elif model_type == "xgb":
        base_estimator = XGBRegressor(random_state=42)
    else:
        return {"error": f"Unknown model_type: {model_type}"}

    # --- Train model ---
    model = MultiOutputRegressor(base_estimator)
    model.fit(X_train_scaled, y_train_scaled)

    # --- Predict ---
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # --- Evaluate ---
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

    # --- Feature importance ---
    feature_names = X_train.columns.tolist()
    importance = np.zeros(len(feature_names))
    try:
        if model_type in ["rf", "xgb", "catb"]:
            all_importances = [est.feature_importances_ for est in model.estimators_]
            importance = np.mean(all_importances, axis=0)
        elif model_type in ["linear", "sgd"]:
            all_coefs = [np.abs(est.coef_) for est in model.estimators_]
            importance = np.mean(all_coefs, axis=0)
    except Exception:
        pass

    if importance.sum() > 0:
        importance = importance / importance.sum() * 100

    feature_importance = [
        {"feature": f, "importance": round(float(imp), 2)}
        for f, imp in zip(feature_names, importance)
    ]

    # --- Map level to dataset column ---
    level_column_map = {"province": "Province", "district": "District"}
    location_col = level_column_map.get(level)

    if location_col not in test_df.columns:
        return {"error": f"Column '{location_col}' not found in dataset"}

    locations = test_df[location_col].values

    # --- Return result ---
    return {
        "level": level,
        "filename": file.filename,
        "model_type": model_type,
        "targets": targets,
        "train_years": train_years,
        "test_years": test_years,
        "mse": mse.tolist(),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "feature_importance": feature_importance,
        "y_true_vs_pred": [
            {
                "location": loc,
                "predictions": y_pred[i].tolist(),
                "y_test": y_test[i].tolist()
            }
            for i, loc in enumerate(locations[:len(y_pred)])  # ป้องกัน index out of range
        ],
    }
