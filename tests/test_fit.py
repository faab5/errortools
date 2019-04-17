import errortools
import numpy as np
import scipy.stats
import pytest

np.random.seed(42)

p_true = np.array([1, 0, -0.25])
b_true = 1.
ndata = 1000

X = np.random.uniform(low=-1, high=1, size=len(p_true) * ndata).reshape((ndata, len(p_true)))
y = (scipy.stats.logistic.cdf(np.dot(X, p_true) + b_true) > np.random.uniform(size=ndata)).astype(int)

@pytest.fixture
def amodel():
    model = errortools.LogisticRegression(fit_intercept=True, l1=0, l2=0)
    model.fit(X, y, initial_parameters=0)
    model.predict(X)
    return model

def test_errors(amodel):
    amodel.estimate_errors(X)
    assert True

def test_errors_sampling(amodel):
    amodel.estimate_errors_sampling(X, 1000)
    assert True

def test_errors_linear(amodel):
    amodel.estimate_errors_linear(X, 1)
    assert True

def test_parameter_limits(amodel):
    limits = [None, (None, -1), (1, None), (5,10)]
    amodel.fit(X, y, initial_parameters=0, parameter_limits=limits)
    p = amodel.parameters
    assert p[1]<-1
    assert p[2]>1
    assert p[3]>5 and p[3]<10

def test_parameter_fixes(amodel):
    limits = [True, False, False, True]
    amodel.fit(X, y, initial_parameters=10, parameter_fixes=limits)
    p = amodel.parameters
    assert p[0] == 10
    assert p[3] == 10