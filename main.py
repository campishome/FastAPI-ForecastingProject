"""
from fastapi import FastAPI, UploadFile, File ,Form
import pandas as pd
import io
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()
#ต่อ DB ได้หากจะเพิ่มในอนาคต

@app.get("/")
def root():
    return {"Message":"Hello world!"}

@app.get("/started")
def get_started():
    return {"Message":"Started"}

@app.post("/upload_excel/")
async def upload_excel(file: UploadFile = File(...)):
    # เช็คว่าเป็นไฟล์ .xlsx หรือไม่
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are supported"}

    # อ่านไฟล์ด้วย pandas
    contents = await file.read()  
    # ใช้ BytesIO เพื่อให้ pandas อ่านจาก memory ได้
    df = pd.read_excel(io.BytesIO(contents))

    # ส่ง 5 แถวแรกกลับ (แปลงเป็น dict)
    head_data = df.head().to_dict(orient="records")

    return {
        "filename": file.filename,
        "head": head_data
    }

@app.post("/upload_and_forecast/")
async def upload_and_forecast(file: UploadFile = File(...)):
    #เช็คว่าไฟล์
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are supported"}

    #DataFrame
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    test_years = [55, 66]

    #Train/Test split
    train_data = df[~df['Year'].isin(test_years)]
    test_data = df[df['Year'].isin(test_years)]

    #Features & Targets
    features = ['Product/ Yield (kg)', 'Duration-month', 'Cutivated Area (rai)', 
                'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg', 
                'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst']
    targets = ['Yield/Harvest-area', 'Yield/Plant-area']

    X_train = train_data[features]
    y_train = train_data[targets]

    X_test = test_data[features]
    y_test = test_data[targets]

    #Train model
    base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_estimator)
    model.fit(X_train, y_train)

    #Predict
    y_pred = model.predict(X_test)

    #Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

    return {
        "filename": file.filename,
        "RMSE": rmse.tolist(),
        "MSE": mse.tolist(),
        "MAE": mae.tolist()
    }
##python -m uvicorn main:app --reload


@app.post("/upload_and_choose_province/")
async def upload_and_forecast_province(Province: str = Form(...) ,file: UploadFile = File(...)):
    #เช็คว่าไฟล์
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are supported"}
    
    # อ่านไฟล์ด้วย pandas
    contents = await file.read()  
    # ใช้ BytesIO เพื่อให้ pandas อ่านจาก memory ได้
    df = pd.read_excel(io.BytesIO(contents))

    #filter เฉพาะจังหวัดที่ผู้ใช้เลือก
    if "Province" not in df.columns:
        return {"error": "Column 'Province' not found in Excel"}

    filtered_df = df[df["Province"] == Province]

    #ส่งออกเป็น dict
    data = filtered_df.to_dict(orient="records")

    return {
        "Province": Province,
        "count": len(filtered_df),
        "list": data
    }
    
#model + rice_type + 


# @app.post("/upload_chooseProvince_and_Forecasting/")
# async def upload_chooseProvince_and_Forecasting(
#     Province: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     #เช็ค
#     if not file.filename.endswith(".xlsx"):
#         return {"error": "Only .xlsx files are supported"}

#     contents = await file.read()
#     df = pd.read_excel(io.BytesIO(contents))

#     #filter เฉพาะจังหวัด
#     if "Province" not in df.columns:
#         return {"error": "Column 'Province' not found in Excel"}

#     filtered_df = df[df["Province"] == Province]

#     if filtered_df.empty:
#         return {"error": f"No data found for Province '{Province}'"}

#     #Train/Test split
#     test_years = [55, 66]  #---* ใช้ปีสุดท้าย
#     train_data = filtered_df[~filtered_df['Year'].isin(test_years)]
#     test_data = filtered_df[filtered_df['Year'].isin(test_years)]

#     #Features & Targets
#     features = [
#         'Product/ Yield (kg)', 'Duration-month', 'Cutivated Area (rai)', 
#         'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg', 
#         'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst'
#     ]
#     targets = ['Yield/Harvest-area', 'Yield/Plant-area']

#     X_train = train_data[features]
#     y_train = train_data[targets]

#     X_test = test_data[features]
#     y_test = test_data[targets]

#     #Train model ---------------------------------------------- rf , sgd , linear regression
#     base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
#     model = MultiOutputRegressor(base_estimator)
#     model.fit(X_train, y_train)

#     #Predict
#     y_pred = model.predict(X_test)

    
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))
#     mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
#     mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

#     return {
#         "filename": file.filename,
#         "Province": Province,
#         "RMSE": rmse.tolist(),
#         "MSE": mse.tolist(),
#         "MAE": mae.tolist()
#     }

#หลังบ้านต้องใช้เป็น saved modle?  *ใช่  *model เรา *มี train + test 
#
#model  พยากรณ์ปีสุดท้าย  พันธ์ข้าว
#ใสไฟล์ excel เข้ามา
#standardization ก่อนจะ forecast  แต่ผลลัพท์ไม่ต้องทำ -----done


model06_paths = {
    "rf": "assets/models/randomforest_model_06.pkl", #กข 6 / อำเภอ
    "sgd": "assets/models/sgd_model_06.pkl", #กข 6 / อำเภอ
    "linear": "assets/models/linearreg_model_06.pkl", #กข 6 / อำเภอ
    "catb": "assets/models/catboost_model_06.pkl",
    "xgb": "assets/models/xgboost_model_06.pkl",
}
model105_paths = {
    "rf": "assets/models/randomforest_model_105.pkl", #ขาวดอกมะลิ105 / อำเภอ
    "sgd": "assets/models/sgd_model_105.pkl", 
    "linear": "assets/models/linearreg_model_105.pkl",
    "catb": "assets/models/catboost_model_105.pkl",
    "xgb": "assets/models/xgboost_model_105.pkl",
}


@app.post("/upload_chooseDistrict_and_Forecasting_06/") 
async def upload_chooseDistrict_and_Forecasting_06(
    District: str = Form(...),
    model_type: str = Form(...),   # ให้เลือก rf / sgd / linear / catb /xgb
    file: UploadFile = File(...)
):
    # ตรวจสอบไฟล์
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are supported"}
    scaler_X = joblib.load("assets/models/scaler_X_06.pkl") #กข 6 / อำเภอ
    scaler_y = joblib.load("assets/models/scaler_y_06.pkl") #กข 6 / อำเภอ
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    # filter เฉพาะจังหวัด
    if "District" not in df.columns:
        return {"error": "Column 'District' not found in Excel"}

    filtered_df = df[df["District"] == District]
    if filtered_df.empty:
        return {"error": f"No data found for District '{District}'"}

    # Features & Targets
    features = [
        'Product/ Yield (kg)', 'Plant-month', 'Cutivated Area (rai)', 
        'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg', 
        'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst'
    ]
    targets = ['Yield/Harvest-area', 'Yield/Plant-area']

    X = filtered_df[features]
    y = filtered_df[targets]

    # Scaling
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    # โหลดโมเดลที่เลือก
    if model_type not in model06_paths:
        return {"error": f"Invalid model_type '{model_type}'. Choose from {list(model06_paths.keys())}"}
    
    model = joblib.load(model06_paths[model_type])

    # ทำนาย
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y, y_pred, multioutput="raw_values"))
    mse = mean_squared_error(y, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y, y_pred, multioutput="raw_values")

    # Result
    results = filtered_df.copy()
    results["Pred_Yield/Harvest-area"] = y_pred[:, 0]
    results["Pred_Yield/Plant-area"] = y_pred[:, 1]
    return {
        "filename": file.filename,
        "District": District,
        "Model": model_type,
        "RMSE": rmse.tolist(),
        "MSE": mse.tolist(),
        "MAE": mae.tolist(),
        "Predictions": results.to_dict(orient="records")
    }


#--- เพิ่มของ 105 / อำเภอ8

@app.post("/upload_chooseDistrict_and_Forecasting_105/") 
async def upload_chooseDistrict_and_Forecasting_105(
    District: str = Form(...),
    model_type: str = Form(...),   # ให้เลือก rf / sgd / linear / catb /xgb
    file: UploadFile = File(...)
):
    # ตรวจสอบไฟล์
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only .xlsx files are supported"}

    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    
    scaler_X = joblib.load("assets/models/scaler_X_105.pkl") #105 / อำเภอ
    scaler_y = joblib.load("assets/models/scaler_y_105.pkl") 
    # filter เฉพาะจังหวัด
    if "District" not in df.columns:
        return {"error": "Column 'District' not found in Excel"}

    filtered_df = df[df["District"] == District]
    if filtered_df.empty:
        return {"error": f"No data found for District '{District}'"}

    # Features & Targets
    features = [
        'Product/ Yield (kg)', 'Plant-month', 'Cutivated Area (rai)', 
        'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg', 
        'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst'
    ]
    targets = ['Yield/Harvest-area', 'Yield/Plant-area']

    X = filtered_df[features]
    y = filtered_df[targets]

    # Scaling
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    # โหลดโมเดลที่เลือก
    if model_type not in model105_paths:
        return {"error": f"Invalid model_type '{model_type}'. Choose from {list(model105_paths.keys())}"}
    
    model = joblib.load(model105_paths[model_type])

    # ทำนาย
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y, y_pred, multioutput="raw_values"))
    mse = mean_squared_error(y, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y, y_pred, multioutput="raw_values")

    # Result
    results = filtered_df.copy()
    results["Pred_Yield/Harvest-area"] = y_pred[:, 0]
    results["Pred_Yield/Plant-area"] = y_pred[:, 1]
    return {
        "filename": file.filename,
        "District": District,
        "Model": model_type,
        "RMSE": rmse.tolist(),
        "MSE": mse.tolist(),
        "MAE": mae.tolist(),
        "Predictions": results.to_dict(orient="records")
    }
"""

from fastapi import FastAPI, Form, File, UploadFile
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = FastAPI()

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

features = [
    'Product/ Yield (kg)', 'Plant-month', 'Cutivated Area (rai)',
    'Harvest-area (rai)', 'rain-avg', 'press-avg', 'temp-avg',
    'RH-avg', 'wind-avg', 'NDVI', 'runoff', 'RootMoist_inst'
]
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

    X = filtered_df[features]

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
        "Forecast": results[[col_name] + targets].to_dict(orient="records")
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

    X = filtered_df[features]
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

    return {
        "filename": file.filename,
        col_name: area_value,
        "Crop": crop,
        "Model": model_type,
        "RMSE": rmse.tolist(),
        "MSE": mse.tolist(),
        "MAE": mae.tolist(),
        "y_true_vs_pred": [
            {"true": yt, "pred": yp} for yt, yp in zip(y.values.tolist(), y_pred.tolist())
        ]
    }

##python -m uvicorn main:app --reload
#vercel --prod