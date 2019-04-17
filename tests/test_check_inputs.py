import errortools
import numpy as np

np.random.seed(42)
estimator = errortools.LogisticRegression()

def test_all_inputs_as_arrays():
    X = np.random.uniform(size=4 * 3).reshape(4,3)
    p = np.random.uniform(size=3)
    y = np.random.choice([0,1], size=4)
    Xp, yp, wp = estimator._check_inputs(X, y, p, False)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

def test_weights_as_number_no_bias():
    X = np.random.uniform(size=4 * 3).reshape(4, 3)
    p = 0
    y = np.random.choice([0,1], size=4)
    Xp, yp, wp = estimator._check_inputs(X, y, p, False)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

def test_weights_as_number_with_bias():
    X = np.random.uniform(size=4 * 3).reshape(4, 3)
    p = 0
    y = np.random.choice([0,1], size=4)
    Xp, yp, wp = estimator._check_inputs(X, y, p, True)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

def test_one_data_point():
    X = np.random.uniform(size=3)
    p = np.random.uniform(size=3)
    y = 1
    Xp, yp, wp = estimator._check_inputs(X, y, p, False)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

def test_one_data_point_and_weight_as_number():
    X = np.random.uniform(size=3)
    p = 0
    y = 1
    Xp, yp, wp = estimator._check_inputs(X, y, p, True)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

def test_features_as_1D_array():
    X = np.random.uniform(size=3)
    p = 0
    y = np.random.choice([0,1], size=3)
    Xp, yp, wp = estimator._check_inputs(X, y, p, True)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

def test_no_targets_no_bias():
    X = np.random.uniform(size=2*3).reshape(2,3)
    p = np.random.uniform(size=3)
    y = None
    Xp, _, wp = estimator._check_inputs(X, y, p, False)
    assert wp.shape[0] == Xp.shape[1]

def test_no_target_and_features_as_1D_array():
    X = np.random.uniform(size=3)
    p = np.random.uniform(size=4)
    y = None
    Xp, _, wp= estimator._check_inputs(X, y, p, True)
    assert wp.shape[0] == Xp.shape[1]

def test_no_target_and_features_as_1D_array_and_weights_as_number():
    X = np.random.uniform(size=3)
    p = 0
    y = None
    Xp, _, wp = estimator._check_inputs(X, y, p, True)
    assert wp.shape[0] == Xp.shape[1]