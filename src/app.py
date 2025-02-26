from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from logger import Logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
import traceback
import sys
import pickle
import configparser
import argparse

SHOW_LOG = True
logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

config = configparser.ConfigParser()
config.read("config.ini")

app = FastAPI()

log.info('FastAPI app initialized')

parser = argparse.ArgumentParser(description="Web App Model")
parser.add_argument("-m", "--model",
                    type=str,
                    help="Select model",
                    required=True,
                    default="LOG_REG",
                    const="LOG_REG",
                    nargs="?",
                    choices=["LOG_REG", "RAND_FOREST", "KNN", "GNB", "D_TREE"])
args = parser.parse_args()

try:
    classifier = pickle.load(open(config[args.model]["path"], "rb"))
except FileNotFoundError:
    log.error(traceback.format_exc())
    sys.exit(1)

scaler = StandardScaler()

log.info('web service model initialized')


@app.get("/")
async def root():
    return {"message": "Hello World"}
    
class PredictionInput(BaseModel):
    X: List[Dict[str, float]]
    y: List[Dict[str, float]]

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        X = scaler.transform(pd.json_normalize(input_data.X))
        y = pd.json_normalize(input_data.y)
        score = classifier.score(X, y)
        pred = classifier.predict(X)
        return {"prediction": pred, "score": score}
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
