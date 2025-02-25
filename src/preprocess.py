import configparser
import os
import pandas as pd

from logger import Logger

SHOW_LOG = True


class DataMaker():
    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.config = configparser.ConfigParser()
        
        self.project_data_path = ".\data"
        
        self.zip_data_path = os.path.join(self.project_data_path, "fashion-mnist.zip")

        self.train_data_path = os.path.join(self.project_data_path, "fashion-mnist_train.csv")
        self.test_data_path = os.path.join(self.project_data_path, "fashion-mnist_test.csv")
       
        self.log.info("DataMaker is ready")

    def split_data_labels(self, data_path):
        dataset = pd.read_csv(data_path)
        
        X = pd.DataFrame(dataset.iloc[:, 1:].values)
        y = pd.DataFrame(dataset.iloc[:, 0].values)

        filename, file_extension = os.path.splitext(data_path)
        subset_mode = filename.split('_')[-1]
        
        X_path = f"{filename}_X{file_extension}"
        y_path = f"{filename}_y{file_extension}"

        X.to_csv(X_path, index=True)
        y.to_csv(y_path, index=True)
        if os.path.isfile(X_path) and os.path.isfile(y_path):
            self.log.info(f"{subset_mode} X and y data is ready")
            self.config["SPLIT_DATA"].update({f'X_{subset_mode}': X_path,
                                              f'y_{subset_mode}': y_path})
            return os.path.isfile(X_path) and os.path.isfile(y_path)
        else:
            self.log.error(f"{subset_mode} X and y data is not ready")
            return False

    def split_data(self) -> bool:
        import zipfile
        with zipfile.ZipFile(self.zip_data_path, 'r') as zip_ref:
            zip_ref.extractall(self.project_data_path)

        self.config["SPLIT_DATA"] = {}

        self.split_data_labels(self.train_data_path)
        self.split_data_labels(self.test_data_path)

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()