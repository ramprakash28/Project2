import csv
import numpy as np
from model.GradientBoostingClassifier import GradientBoostingClassifier

def test_predict_regular():
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)

    data = []
    with open("tests/test_data_regular.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('feature')] for datum in data])
    y = np.array([float(datum['target']) for datum in data])

    model.fit(X, y)
    preds = model.predict(X)

    assert all(pred in [0.0, 1.0] for pred in preds), "Predictions are not binary!"
