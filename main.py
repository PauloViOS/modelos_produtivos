import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

model = joblib.load('trained_logistic_regression_model.pkl')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    input: list[float]


@app.post('/predict')
async def predict(input_data: InputData):
    data = np.array(input_data.input)
    print(data)
    prediction = model.predict(data.reshape(1, -1))
    print(prediction)
    return {'prediction': prediction.tolist()}
