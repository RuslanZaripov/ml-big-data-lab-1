import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from preprocess import DataMaker

config = configparser.ConfigParser()
config.read("config.ini")


class TestDataMaker(unittest.TestCase):

    def setUp(self) -> None:
        self.data_maker = DataMaker()

    def test_split_data(self):
        self.assertEqual(self.data_maker.split_data(), True)

    def test_split_data_labels(self):
        X_test, y_test = self.data_maker.split_data_labels(config["DATA"]["test"])
        self.assertEqual(os.path.isfile(X_test) and os.path.isfile(y_test), True)


if __name__ == "__main__":
    unittest.main()