import configparser
import os
import unittest
import sys
from train import Model


sys.path.insert(1, os.path.join(os.getcwd(), "src"))

config = configparser.ConfigParser()
config.read("config.ini")

class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Model()

    def test_log_reg(self):
        self.assertEqual(self.model.log_reg(use_config=False), True)


if __name__ == "__main__":
    unittest.main()