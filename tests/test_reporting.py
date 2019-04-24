import errortools
import numpy as np
import scipy.stats
import pytest
from matplotlib.backends.backend_pdf import PdfPages
import os

np.random.seed(42)

p_true = np.array([1, 0, -0.25])
b_true = 1.
ndata = 1000

X = np.random.uniform(low=-1, high=1, size=len(p_true) * ndata).reshape((ndata, len(p_true)))
y = (scipy.stats.logistic.cdf(np.dot(X, p_true) + b_true) > np.random.uniform(size=ndata)).astype(int)

features = ['x1', 'x2', 'x3', 'bias']

@pytest.fixture
def amodel():
    model = errortools.LogisticRegression(fit_intercept=True, l1=0, l2=0)
    model.fit(X, y, initial_parameters=0)
    model.predict(X)
    
    return model

def test_report_correlation(amodel):
    with PdfPages('Report.pdf') as pdf:
        errortools.report_correlation_matrix(amodel, pdf, features)
        
    assert os.path.isfile('Report.pdf') == True
    assert os.path.isfile('Reports.pdf') == False

def test_report_parameter_error(amodel):
    with PdfPages('Report.pdf') as pdf:
        errortools.report_parameter_error(amodel, pdf, features)
        
    assert os.path.isfile('Report.pdf') == True
    assert os.path.isfile('Reports.pdf') == False

def test_report_loss_versus_approximation(amodel):
    with PdfPages('Report.pdf') as pdf:
        errortools.report_loss_versus_approximation(amodel, X, y, 0, 0, features, pdf)

    assert os.path.isfile('Report.pdf') == True
    assert os.path.isfile('Reports.pdf') == False