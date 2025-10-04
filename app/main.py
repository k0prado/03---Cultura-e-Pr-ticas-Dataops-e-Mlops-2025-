import os
import json
import mlflow
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI

class fetalHeathData(BaseModel):
    accelerations = float
    fetal_movement = float
    uterine_contractions = float
    severe_decelerations = float


app = FastAPI(title="Fetal Health Api",
            openapi_tags=[
            {
                "name":"Health",
                "description":"Get api heath"
            },
            {
                "name":"Prediction",
                "description":"Model prediction"
            }
        ])

def load_model():
    MLFLOW_TRACKING_URI = ''
    MLFLOW_TRACKING_USERNAME = ''
    MLFLOW_TRACKING_PASSWORD = ''
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_model = client.get_registered_model('fatal_health')
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

@app_os_event(event_type="startup")
def startup_event():
    global load_model
    load_model = load_model()


@app.get("/", tags=["Health"])
def api_health():
    return {"status": "healthy"}

@app.post("/predict", tags=["Prediction"])
def predict(request: fetalHeathData):
    global load_model

    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)

    prediction = loaded_model.predict(received_data)

    return {"prediction": str(prediction[0])}