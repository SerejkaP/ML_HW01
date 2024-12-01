import pickle
from enum import Enum

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline
from fastapi.responses import StreamingResponse
import io

# Импортирую, чтобы модель заработала. Ей нужны эти трансформеры
from transformers import NumericalTransformer, CategoryTransformer, CarNameToMaker, StrToNum


class Fuel(str, Enum):
    diesel = 'Diesel',
    petrol = 'Petrol',
    lpg = 'LPG',
    cng = 'CNG'

class SellerType(str, Enum):
    individual = 'Individual',
    dealer = 'Dealer',
    trustmark_dealer= 'Trustmark Dealer'

class Transmission(str, Enum):
    manual = 'Manual',
    automatic = 'Automatic'

class Owner(str, Enum):
    first = 'First Owner',
    second = 'Second Owner',
    third = 'Third Owner',
    fourth_or_above = 'Fourth & Above Owner',
    test_drive_car = 'Test Drive Car'

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: Fuel
    seller_type: SellerType
    transmission: Transmission
    owner: Owner
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


app = FastAPI()

model: Pipeline

@app.on_event("startup")
def load_model():
    with open("model.pickle", "rb") as f:
        global model
        model = pickle.load(f)

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame([item.dict()])
    return model.predict(data)


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> StreamingResponse:
    try:
        contents = file.file.read()
        file_stream = io.BytesIO(contents)
        data = pd.read_csv(file_stream)
        selling_prices = model.predict(data)
        data['selling_price'] = selling_prices

        stream = io.StringIO()
        data.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        return response
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
