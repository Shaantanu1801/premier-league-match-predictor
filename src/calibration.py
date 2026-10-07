import numpy as np
from sklearn.linear_model import LogisticRegression


class VectorCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(C=10.0, max_iter=2000)

    def fit(self, p, y):
        self.lr.fit(np.log(np.clip(p, 1e-6, 1)), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(np.log(np.clip(p, 1e-6, 1)))


