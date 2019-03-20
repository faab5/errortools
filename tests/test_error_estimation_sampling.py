import errortools
import numpy as np
import scipy.stats

np.random.seed(42)

def test_estimate_errors_sampling():
    w_true = np.array([1, 0, -0.25])
    b_true = 1.
    ndata = 1000

    X = np.random.uniform(low=-1, high=1, size=len(w_true) * ndata).reshape((ndata, len(w_true)))
    y = (scipy.stats.logistic.cdf(np.dot(X, w_true) + b_true) > np.random.uniform(size=ndata)).astype(int)

    model = errortools.LogisticRegression(fit_intercept=True, l1=0, l2=0)
    model.fit(X, y, w0=0)
    el_10, _ = model.estimate_errors_sampling(X, 10)
    el_1000, _ = model.estimate_errors_sampling(X, 1000)
    
    #Compare error to different estimation method
    el_1, _ = model.estimate_errors(X, 1)
    diff_few_samples = el_1.mean() - el_10.mean()
    diff_many_samples = el_1.mean() - el_1000.mean()
    
    assert diff_few_samples > diff_many_samples