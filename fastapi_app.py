from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import re
import pandas as pd
import io

app = FastAPI()

# Загрузка моделей из pickle файла
with open("Lasso.pkl", "rb") as f:
    pickle_object = pickle.load(f)
    model = pickle_object["Lasso_model"]
    scaler = pickle_object["scaler"]
    

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def handle_input(df):
    # отфильтруем датафрейм поданный на вход - по сути то же, что было в домашке в Ноутбуке
    df['mileage'] = df['mileage'].apply(lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0]) if pd.notnull(x) else None)
    df['engine'] = df['engine'].apply(lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0]) if pd.notnull(x) else None)
    df['max_power'] = df['max_power'].apply(lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0]) if pd.notnull(x) and re.findall(r'\d+\.?\d*', str(x)) else None)
    df = df.astype({
        'year': 'float32',
        'km_driven': 'float32',
        'mileage': 'float32',
        'engine': 'int64',
        'max_power': 'float32',
        'seats': 'float32'
    }) # приведем типы от строковых к числовым чтобы scaler работал корректно
    df = df.drop('selling_price', axis=1)
    #отбор только числовых столбцов
    df = df.select_dtypes(include='number')
    #scaling 
    df=scaler.transform(df)
    df = pd.DataFrame(data=df)
    print(df)
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = np.array([[item.name, item.year, item.selling_price, item.km_driven, item.fuel, item.seller_type, item.transmission, 
                      item.owner, item.mileage, item.engine, item.max_power, item.torque, item.seats]])
    data = pd.DataFrame(data, columns=['name','year','selling_price','km_driven', 'fuel', 'seller_type', 'transmission', 
                                       'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats'])
    data = pd.DataFrame(data)
    data = handle_input(data)
    prediction = model.predict(data)
    return prediction[0]


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> StreamingResponse:
    try:
        df = pd.read_csv(file.file)
        predictions = model.predict(df)
        df['predicted_price'] = predictions
        # Преобразование DataFrame в CSV
        stream = io.StringIO() # буфер для хранения данных
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=512, detail=str(e))