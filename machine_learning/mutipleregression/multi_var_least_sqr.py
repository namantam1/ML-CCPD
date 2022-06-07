import numpy as np
from numpy import ndarray
from numpy.linalg import inv
from sklearn.metrics import r2_score

class LeastSqr:
    @property
    def coef_(self):
        return self.A[1:]

    @property
    def intercept_(self):
        return self.A[0]

    def score(self, X: ndarray, y: ndarray):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def predict(self, x: ndarray):
        x = np.column_stack((np.ones(x.shape[0]), x))

        return x @ self.A

    def fit(self, X: ndarray, y: ndarray):
        X = np.column_stack((np.ones(X.shape[0]), X))
        _, n = X.shape

        sx = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                sx[i, j] = (X[:, i]*X[:, j]).sum()

        sy = np.zeros(n)
        for i in range(n):
            sy[i] = (X[:, i] * y).sum()

        self.A = inv(sx) @ sy

# x = array([x]).T