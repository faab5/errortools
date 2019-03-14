import errortools
import numpy as np
import scipy.stats

np.random.seed(42)

w_true = np.array([1, 0, -0.25])
w_true_normed = w_true/np.sqrt(np.sum(w_true**2))
ndata = 1000

X = np.random.uniform(low=-1, high=1, size=2*ndata).reshape((ndata, 2))
X_with_intercept = np.concatenate((X, np.ones((ndata,1), dtype=float)), axis=1)
y = (scipy.stats.logistic.cdf(np.dot(X_with_intercept, w_true)) > np.random.uniform(size=ndata)).astype(int)


model = errortools.LogisticRegression(fit_intercept=True, l1=0, l2=0)
model.fit(X, y, w0=0)
model.predict(X)
model.estimate_errors(X)