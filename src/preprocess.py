import configparser
import os
import pandas as pd
import zipfile

from logger import Logger

SHOW_LOG = True


class DataMaker():
    """
    A class to handle the preparation and preprocessing of the Fashion-MNIST dataset.

    This class is responsible for unzipping the dataset, splitting it into training and testing sets. 
    It also saves the paths to these files in a configuration file for easy access.
    """

    def __init__(self) -> None:
        """
        Initializes the DataMaker class.

        Sets up the logger, configuration parser, and defines the paths for the dataset files.
        """
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.config = configparser.ConfigParser()
        
        self.project_data_path = os.path.join(os.getcwd(), "data")
        
        self.zip_data_path = os.path.join(self.project_data_path, "fashion-mnist.zip")

        self.train_data_path = os.path.join(self.project_data_path, "fashion-mnist_train.csv")
        self.test_data_path = os.path.join(self.project_data_path, "fashion-mnist_test.csv")

        self.config['DATA'] = {'train': self.train_data_path,
                               'test': self.test_data_path}
       
        self.log.info("DataMaker is ready")

    def split_data_labels(self, data_path):
        """
        Splits the dataset into features (X) and labels (y).

        Args:
            data_path (str): Path to the dataset CSV file.

        Returns:
            tuple: Paths to the saved features (X) and labels (y) CSV files.
        """
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
        else:
            self.log.error(f"{subset_mode} X and y data is not ready")

        return X_path, y_path

    def split_data(self) -> bool:
        """
        Preprocesses the dataset by unzipping it and splitting it into training and testing sets.

        Returns:
            bool: True if all files were successfully created, False otherwise.
        """
        self.log.info('Start preprocessing...')

        with zipfile.ZipFile(self.zip_data_path, 'r') as zip_ref:
            zip_ref.extractall(self.project_data_path)

        self.log.info(f"Unzip {self.zip_data_path} done")

        X_train, y_train = self.split_data_labels(self.train_data_path)
        X_test, y_test = self.split_data_labels(self.test_data_path)
        
        self.config["SPLIT_DATA"] = {'X_train': X_train,
                                     'X_test': X_test,
                                     'y_train': y_train,
                                     'y_test': y_test}

        self.log.info(f"Data preprocessing done")

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

        return all([os.path.exists(path) 
                    for path in self.config['SPLIT_DATA'].values()])

if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()
