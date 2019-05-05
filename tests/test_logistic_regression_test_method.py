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

X_test = np.random.uniform(low=-1, high=1, size=len(p_true) * ndata).reshape((ndata, len(p_true)))
y_test = (scipy.stats.logistic.cdf(np.dot(X, p_true) + b_true) > np.random.uniform(size=ndata)).astype(int)

@pytest.fixture
def amodel():
    model = errortools.LogisticRegression(l1=0, l2=0)
    model.fit(X, y, initial_parameters=0)
    return model

def test_logistic_regression_test_method(amodel):
    d_out = amodel.test(X_test, y_test)
    assert 'auc' in d_out.keys()
    assert 'value' in d_out['auc'].keys()
    assert 'error' in d_out['auc'].keys()
    assert 'confusion matrix' in d_out.keys()
    assert 'true positives' in d_out['confusion matrix'].keys()
    assert 'value' in d_out['confusion matrix']['true positives'].keys()
    assert 'error' in d_out['confusion matrix']['true positives'].keys()
    assert 'false negatives' in d_out['confusion matrix'].keys()
    assert 'value' in d_out['confusion matrix']['false negatives'].keys()
    assert 'error' in d_out['confusion matrix']['false negatives'].keys()
    assert 'true negatives' in d_out['confusion matrix'].keys()
    assert 'value' in d_out['confusion matrix']['true negatives'].keys()
    assert 'error' in d_out['confusion matrix']['true negatives'].keys()
    assert 'false positives' in d_out['confusion matrix'].keys()
    assert 'value' in d_out['confusion matrix']['false positives'].keys()
    assert 'error' in d_out['confusion matrix']['false positives'].keys()
    assert 'cm_cvr_mtx' in d_out['confusion matrix'].keys()
    assert d_out['confusion matrix']['cm_cvr_mtx'].shape == (4,4)
    assert 'prediction rates' in d_out.keys()
    assert 'true positive rate' in d_out['prediction rates'].keys()
    assert 'value' in d_out['prediction rates']['true positive rate'].keys()
    assert 'error' in d_out['prediction rates']['true positive rate'].keys()
    assert 'true negative rate' in d_out['prediction rates'].keys()
    assert 'value' in d_out['prediction rates']['true negative rate'].keys()
    assert 'error' in d_out['prediction rates']['true negative rate'].keys()
    assert 'covariance' in d_out['prediction rates'].keys()