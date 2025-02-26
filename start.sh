#!/bin/bash

python src/preprocess.py
python src/train.py
python src/predict.py -m LOG_REG -t func

coverage run src/unit_tests/test_preprocess.py
coverage run -a src/unit_tests/test_training.py
coverage report -m

fastapi run src/app.py --port 8000 & sleep 5

curl -X GET http://localhost:8000/
