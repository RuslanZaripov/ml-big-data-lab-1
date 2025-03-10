import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
import shutil
import sys
import time
import traceback
import yaml

from logger import Logger

SHOW_LOG = True


class Predictor():
    """
    A class used to handle model prediction and testing.

    This class loads a trained model, performs smoke or functional tests, and logs the results.
    It also saves experiment configurations and logs for further analysis.
    """

    def __init__(self, args) -> None:
        """
        Initializes the Predictor class.

        Args:
            args (argparse.Namespace): Command-line arguments specifying the model and test type.
        """
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        self.args = args
        
        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
        
        try:
            self.sc = pickle.load(open(self.config["STD_SCALER"]["path"], "rb"))
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        """
        Performs prediction and testing based on the specified model and test type.

        This method loads the specified model, performs either smoke or functional tests,
        and logs the results. For functional tests, it also saves experiment configurations
        and logs for further analysis.

        Returns:
            bool: True if the prediction and testing process completes successfully.
        """
        try:
            classifier = pickle.load(open(self.config[self.args.model]["path"], "rb"))
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
            
        if self.args.tests == "smoke":
            try:
                score = classifier.score(self.X_test, self.y_test)
                print(f'{self.args.model} has {score} score')
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)

            self.log.info(f'{self.config[self.args.model]["path"]} passed smoke tests')
            
        elif self.args.tests == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")

            for test in os.listdir(tests_path):
                json_path = os.path.join(tests_path, test)
                with open(json_path) as f:
                    try:
                        data = json.load(f)
                        X = self.sc.transform(pd.json_normalize(data, record_path=['X']))
                        y = pd.json_normalize(data, record_path=['y'])
                        score = classifier.score(X, y)
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)

                    self.log.info(f'{self.config[self.args.model]["path"]} passed func test {f.name}')

                    exp_data = {
                        "model": self.args.model,
                        "model params": dict(self.config.items(self.args.model)),
                        "tests": self.args.tests,
                        "score": str(score),
                        "X_test path": self.config["SPLIT_DATA"]["x_test"],
                        "y_test path": self.config["SPLIT_DATA"]["y_test"],
                    }
                    str_date_time = datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir,"exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False) 
                    
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir,"exp_logfile.log"))
                    shutil.copy(self.config[self.args.model]["path"], os.path.join(exp_dir,f'exp_{self.args.model}.sav'))

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor")
        
    parser.add_argument("-m", "--model",
                        type=str,
                        help="Select model",
                        required=True,
                        default="LOG_REG",
                        const="LOG_REG",
                        nargs="?",
                        choices=["LOG_REG"])
    
    parser.add_argument("-t", "--tests",
                        type=str,
                        help="Select tests",
                        required=True,
                        default="smoke",
                        const="smoke",
                        nargs="?",
                        choices=["smoke", "func"])
    
    args = parser.parse_args()

    predictor = Predictor(args)
    predictor.predict()
