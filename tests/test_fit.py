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
w_fit_1 = model.weights
w_fit_1_normed = w_fit_1/np.sqrt(np.sum(w_fit_1**2)+1e-12)

assert np.sqrt(np.sum((w_fit_1_normed-w_true_normed)**2)) < 0.1


model.fit(X, y, w0=0, l2=10)
w_fit_2 = model.weights
w_fit_2_normed = w_fit_2/np.sqrt(np.sum(w_fit_2**2)+1e-12)
print(w_fit_2)
print(w_fit_2_normed)

assert np.sqrt(np.sum((w_fit_2_normed-w_true_normed)**2)) > 0.1


model.fit(X, y, w0=0, l1=10, l2=0)
w_fit_3 = model.weights
w_fit_3_normed = w_fit_3/np.sqrt(np.sum(w_fit_3**2)+1e-12)
print(w_fit_3)
print(w_fit_3_normed)

if np.any(np.isnan(model.weights)):
    raise ValueError(model.weights)
if np.any(np.isnan(model.cvr_mtx)):
    raise ValueError(model.cvr_mtx)

assert np.sqrt(np.sum((w_fit_3_normed-w_true_normed)**2)) > 0.1