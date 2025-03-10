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
import uvicorn

class PredictionInput(BaseModel):
    X: List[Dict[str, float]]
    y: List[Dict[str, float]]

class WebApp:
    def __init__(self, args):
        SHOW_LOG = True
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        self.args = args

        self.model, self.scaler = self._load_model()
        self.log.info('web app model initialized')

        self.app = self._create_app()
        self.log.info('FastAPI app initialized')

        self.prediction_service = None

    def _create_app(self):
        app = FastAPI()

        @app.get("/")
        async def root():
            return {"message": "Hello World"}

        @app.post("/predict")
        async def predict(input_data: PredictionInput):
            try:
                X = self.scaler.transform(pd.json_normalize(input_data.X))
                y = pd.json_normalize(input_data.y)
                score = self.model.score(X, y)
                pred = self.model.predict(X).tolist()
                return {"prediction": pred, "score": score}
            except Exception as e:
                self.log.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

        return app

    def _load_model(self):
        try:
            classifier = pickle.load(open(self.config[self.args.model]["path"], "rb"))
            scaler = pickle.load(open(self.config["STD_SCALER"]["path"], "rb"))
            return classifier, scaler
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
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

    web_app = WebApp(args)
    web_app.run()
