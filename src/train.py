import configparser
import os
import pandas as pd
import pickle
import sys
import traceback
from logger import Logger
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SHOW_LOG = True


class Model():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        
        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0).to_numpy().reshape(-1)
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0).to_numpy().reshape(-1)
        
        self.project_experiments_path = os.path.join(os.getcwd(), "experiments")
        
        self.sc_path = os.path.join(self.project_experiments_path, 'sc.pkl')
        
        sc = StandardScaler()
        try:
            self.X_train = sc.fit_transform(self.X_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        self.X_test = sc.transform(self.X_test)
        self.save_model(sc, self.sc_path, 'STD_SCALER', {'path': self.sc_path})

        self.log_reg_path = os.path.join(self.project_experiments_path, "log_reg.sav")
        
        self.log.info("Model is ready")

    def log_reg(self, use_config: bool, predict=False, max_iter=100) -> bool:
        if use_config:
            try:
                 classifier = LogisticRegression(max_iter=self.config.getint("LOG_REG", "max_iter"))
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
             classifier = LogisticRegression(max_iter=max_iter)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'max_iter': max_iter,
                  'path': self.log_reg_path}
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.log.info(f'{path} is saved')
        
        return os.path.isfile(path)


if __name__ == "__main__":
    model = Model()
    model.log_reg(use_config=False, predict=True)
