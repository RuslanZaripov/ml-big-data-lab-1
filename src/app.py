from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from logger import Logger

SHOW_LOG = True
logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

app = FastAPI()

log.info('FastAPI app Initialized')


@app.get("/")
async def root():
    return {"message": "Hello World"}
    
class PredictionInput(BaseModel):
    X: List[Dict[str, float]]
    y: List[Dict[str, float]]

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        X = input_data.X
        y = input_data.y
        return {"X": X, "y": y}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
