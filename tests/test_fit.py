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

def test_refit(amodel):
    amodel.fit(X, y, initial_parameters=1, initial_step_sizes=1, parameter_limits=[None,(None, 2), (0, None), (-1,2),], parameter_fixes=[True,False,False,True])
    amodel.fit(X, y, parameter_limits=False, parameter_fixes=False)

    with pytest.raises(Exception):
        amodel.fit(X, y, initial_parameters=[2,3,4,'a'])
    with pytest.raises(Exception):
        amodel.fit(X, y, initial_parameters='a')
    with pytest.raises(Exception):
        amodel.fit(X, y, initial_parameters=[1,2,3,4,5])

    with pytest.raises(Exception):
        amodel.fit(X, y, initial_step_sizes=[2,3,4,'a'])
    with pytest.raises(Exception):
        amodel.fit(X, y, initial_step_sizes='a')
    with pytest.raises(Exception):
        amodel.fit(X, y, initial_step_sizes=[1,2,3,4,5])

    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_limits=['a', (0,1), (0,1), (0,1)])
    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_limits=[('a',1), (0,1), (0,1), (0,1)])
    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_limits=[(0,1), (0,1), (0,1), (0,1), (0,1)])
    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_limits='a')

    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_fixes=['a', False, False, False])
    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_fixes=[False, False, False, False, False])
    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_fixes=[False, False, False])
    with pytest.raises(Exception):
        amodel.fit(X, y, parameter_fixes='a')

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