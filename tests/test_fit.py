import errortools
import numpy as np
import scipy

np.random.seed(42)

def test_fit_predict_and_estimate_errors():
    w_true = np.array([1, 0, -0.25])
    b_true = 1.
    ndata = 1000

    X = np.random.uniform(low=-1, high=1, size=len(w_true) * ndata).reshape((ndata, len(w_true)))
    y = (scipy.stats.logistic.cdf(np.dot(X, w_true) + b_true) > np.random.uniform(size=ndata)).astype(int)

    model = errortools.LogisticRegression(fit_intercept=True, l1=0, l2=0)
    model.fit(X, y, w0=0)
    model.predict(X)
    model.estimate_errors(X)
    model.estimate_errors_sampling(X, 1000)
    assert True