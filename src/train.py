import configparser
import os
import pandas as pd
import pickle
import traceback
from logger import Logger
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SHOW_LOG = True


class MultiModel:
    def __init__(self) -> None:
        self.logger = Logger(SHOW_LOG)
        self.log = self.logger.get_logger(__name__)
        self.config = self._load_config()

        self.X_train, self.y_train, self.X_test, self.y_test = self._load_data()
        self.model_paths = self._set_model_paths()
        
        self.log.info("MultiModel is ready")

    def _load_config(self):
        """Loads configuration from config.ini."""
        config = configparser.ConfigParser()
        config.read("config.ini")
        return config

    def _load_data(self):
        """Loads training and testing data, applies feature scaling."""
        X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
        y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0).to_numpy().reshape(-1)
        X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
        y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0).to_numpy().reshape(-1)

        scaler = StandardScaler()
        return scaler.fit_transform(X_train), y_train, scaler.transform(X_test), y_test

    def _set_model_paths(self):
        """Defines model save paths."""
        experiments_path = "./experiments"
        return {
            "LOG_REG": os.path.join(experiments_path, "log_reg.sav"),
            "RAND_FOREST": os.path.join(experiments_path, "rand_forest.sav"),
            "KNN": os.path.join(experiments_path, "knn.sav"),
            "SVM": os.path.join(experiments_path, "svm.sav"),
            "GNB": os.path.join(experiments_path, "gnb.sav"),
            "D_TREE": os.path.join(experiments_path, "d_tree.sav"),
        }

    def train_and_save_model(self, model_name: str, classifier, predict: bool = False) -> bool:
        """
        Generic method to train, evaluate, and save a model.
        """
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            raise RuntimeError(f"Failed to train {model_name}")

        if predict:
            y_pred = classifier.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"{model_name} Accuracy: {accuracy:.4f}")

        return self._save_model(model_name, classifier)

    def _save_model(self, model_name: str, classifier) -> bool:
        """Saves the model and updates the config file."""
        path = self.model_paths[model_name]
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.config[model_name] = {"path": path}
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

        self.log.info(f"{model_name} model saved at {path}")
        return os.path.isfile(path)

    # Model-specific methods
    def log_reg(self, use_config=False, predict=False):
        max_iter = self.config.getint("LOG_REG", "max_iter", fallback=1000) if use_config else 1000
        return self.train_and_save_model("LOG_REG", LogisticRegression(max_iter=max_iter), predict)

    def rand_forest(self, use_config=False, predict=False):
        n_estimators = self.config.getint("RAND_FOREST", "n_estimators", fallback=100)
        criterion = self.config.get("RAND_FOREST", "criterion", fallback="entropy")
        return self.train_and_save_model("RAND_FOREST", RandomForestClassifier(n_estimators=n_estimators, criterion=criterion), predict)

    def knn(self, use_config=False, predict=False):
        n_neighbors = self.config.getint("KNN", "n_neighbors", fallback=5)
        metric = self.config.get("KNN", "metric", fallback="minkowski")
        p = self.config.getint("KNN", "p", fallback=2)
        return self.train_and_save_model("KNN", KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p), predict)

    def svm(self, use_config=False, predict=False):
        kernel = self.config.get("SVM", "kernel", fallback="linear")
        random_state = self.config.getint("SVM", "random_state", fallback=0)
        return self.train_and_save_model("SVM", SVC(kernel=kernel, random_state=random_state), predict)

    def gnb(self, predict=False):
        return self.train_and_save_model("GNB", GaussianNB(), predict)

    def d_tree(self, use_config=False, predict=False):
        criterion = self.config.get("D_TREE", "criterion", fallback="entropy")
        return self.train_and_save_model("D_TREE", DecisionTreeClassifier(criterion=criterion), predict)


if __name__ == "__main__":
    multi_model = MultiModel()
    multi_model.log_reg(predict=True)
    multi_model.rand_forest(predict=True)
    multi_model.knn(predict=True)
    multi_model.gnb(predict=True)
    multi_model.d_tree(predict=True)
