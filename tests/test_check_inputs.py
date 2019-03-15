import errortools
import numpy as np

np.random.seed(42)

def test_logistic_regression_check_inputs():
    estimator = errortools.LogisticRegression()

    # check shapes of X, y, weights w and bias b
    X = np.random.uniform(size=4 * 3).reshape(4,3)
    w = np.random.uniform(size=3)
    y = np.random.choice([0,1], size=4)
    b = 0
    Xp, yp, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

    # check shapes of X, y, weights w and bias b
    # when weights are given as a single number
    # _check_inputs will expand the weights to
    # the right dimension
    X = np.random.uniform(size=4 * 3).reshape(4, 3)
    w = 0
    y = np.random.choice([0,1], size=4)
    Xp, yp, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

    # check shapes of X, y, weights w and bias b
    # when X is given as a 1D vector
    # in stead of 2D matrix
    # _check_inputs will reshape X
    # to fit the target
    X = np.random.uniform(size=3)
    w = np.random.uniform(size=3)
    y = 1
    Xp, yp, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

    # check shapes of X, y, weights w and bias b
    # when X is given as a 1D vector
    # and w as a single number
    # _check_inputs will expand X to match y
    # and expand w to match X
    X = np.random.uniform(size=3)
    w = 0
    y = 1
    Xp, yp, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

    X = np.random.uniform(size=3)
    w = 0
    y = np.random.choice([0,1], size=3)
    Xp, yp, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]
    assert yp.shape[0] == Xp.shape[0]

    # check shapes of X and w
    # when y is not given
    X = np.random.uniform(size=2*3).reshape(2,3)
    w = np.random.uniform(size=3)
    y = None
    Xp, _, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]

    # check shapes of X and w
    # when y is not given
    # and X is given as a 1D array
    X = np.random.uniform(size=3)
    w = np.random.uniform(size=3)
    y = None
    Xp, _, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]

    # check shapes of X and w
    # when y is not given
    # and X is given as a 1D array
    # and w is given as a single number
    X = np.random.uniform(size=3)
    w = 0
    y = None
    Xp, _, wp, bp = estimator._check_inputs(X, y, w, b)
    assert wp.shape[0] == Xp.shape[1]