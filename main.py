from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import re
import os
from pathlib import Path
import uuid

app = FastAPI()

# Конфигурация путей
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "downloads"

# Создаем необходимые директории
STATIC_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Загрузка моделей (с обработкой ошибок)
try:
    models = {
        '20_29': joblib.load(MODELS_DIR / 'model_data_20_29.joblib'),
        '30_39': joblib.load(MODELS_DIR / 'model_data_30_39.joblib'),
        '40_49': joblib.load(MODELS_DIR / 'model_data_40_49.joblib'),
        '50_69': joblib.load(MODELS_DIR / 'model_data_50_69.joblib'),
        '70': joblib.load(MODELS_DIR / 'model_data_70.joblib'),
        'full': joblib.load(MODELS_DIR / 'model_data_full.joblib'),
    }
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки моделей: {str(e)}")

# Загрузка интервалов
try:
    intervals_df = pd.read_excel(MODELS_DIR / "prediction_intervals.xlsx")
    intervals_df['Model'] = intervals_df['Model'].str.lower()
    prediction_intervals = intervals_df.set_index('Model')[['Lower_Bound_95%', 'Upper_Bound_95%']].to_dict(orient='index')
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки интервалов: {str(e)}")

def parse_expected_age(value):
    if isinstance(value, str):
        if '+' in value:
            base = int(value.replace('+', '').strip())
            return (base + 80) / 2
        nums = re.findall(r'\d+', value)
        if len(nums) == 2:
            return (int(nums[0]) + int(nums[1])) / 2
        elif len(nums) == 1:
            return float(nums[0])
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan

@app.get("/")
async def get_homepage():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/download_example/")
async def download_example():
    example_path = STATIC_DIR / "example.xlsx"
    if not example_path.exists():
        raise HTTPException(status_code=404, detail="Example file not found")
    return FileResponse(example_path, filename="example.xlsx")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted")

    try:
        contents = await file.read()
        new_data = pd.read_excel(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    results = []
    for _, row in new_data.iterrows():
        try:
            expected_raw = row.get('expected_age', np.nan)
            expected_age = parse_expected_age(expected_raw)

            if pd.isna(expected_age):
                group = 'full'
            elif expected_age <= 29:
                group = '20_29'
            elif expected_age <= 39:
                group = '30_39'
            elif expected_age <= 49:
                group = '40_49'
            elif expected_age >= 70:
                group = '70'
            else:
                group = '50_69'

            model = models[group]
            interval = prediction_intervals.get(group, {'Lower_Bound_95%': 10, 'Upper_Bound_95%': 10})

            X_sample = row.drop(labels=['ID', 'expected_age'], errors='ignore').to_frame().T
            X_sample_filled = X_sample.apply(lambda row: row.fillna(row.mean()), axis=1)

            y_pred = model.predict(X_sample_filled)[0]
            ci_lower = max(y_pred - interval['Lower_Bound_95%'], 16)
            ci_upper = min(y_pred + interval['Upper_Bound_95%'], 100)

            results.append({
                **row.to_dict(),
                'Predicted_AGE': y_pred,
                'CI_95_lower': ci_lower,
                'CI_95_upper': ci_upper,
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing row: {str(e)}")

    # Сохраняем результаты во временный файл
    output_df = pd.DataFrame(results)
    output_filename = f"predictions_{uuid.uuid4().hex}.xlsx"
    output_path = OUTPUT_DIR / output_filename
    output_df.to_excel(output_path, index=False)

    return FileResponse(path=output_path, filename="predictions.xlsx", media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
