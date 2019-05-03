import errortools
import numpy as np
import pytest
import scipy.stats

np.random.seed(42)

@pytest.fixture
def non_fitted_model():
    model = errortools.LogisticRegression(l1=0, l2=0)
    return model

@pytest.fixture
def fitted_model():
    X = np.random.uniform(low=-1, high=1, size=100*2).reshape((100, 2))
    y = np.random.choice([0,1], size=100)
    model = errortools.LogisticRegression(l1=0, l2=0)
    model.fit(X, y, initial_parameters=0)
    return model



def test_not_fitted_features_and_target_1(non_fitted_model):
    X = np.random.uniform(size=4 * 3).reshape(4, 3)
    y = np.random.choice([0, 1], size=4)
    U, v = non_fitted_model._check_inputs(X, y)
    assert v.shape[0] == U.shape[0]

def test_not_fitted_features_and_target_2(non_fitted_model):
    X = np.random.uniform(size=4)
    y = np.random.choice([0, 1], size=4)
    U, v = non_fitted_model._check_inputs(X, y)
    assert v.shape[0] == U.shape[0]

def test_fitted_features_and_target(fitted_model):
    X = np.random.uniform(size=4*2).reshape(4, 2)
    y = np.random.choice([0, 1], size=4)
    U, v = fitted_model._check_inputs(X, y)
    assert v.shape[0] == U.shape[0]
    assert U.shape[1] + 1 == fitted_model.parameters.shape[0]

def test_fitted_features_no_target(fitted_model):
    X = np.random.uniform(size=2)
    U, v = fitted_model._check_inputs(X, None)
    assert U.shape[1] + 1 == fitted_model.parameters.shape[0]
