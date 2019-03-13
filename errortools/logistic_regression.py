import numpy as np
import iminuit
import scipy.stats

class LogisticRegression():
    """

    """

    def __init__(self):
        pass

    def fit(self, X, y, w0=0, fit_intercept=True):
        pass
    
    def predict(self, X):
        pass
    
    def predict(self, X, w):
        pass
    
    def estimate_errors(self, X):
        pass

    def _check_inputs(self, X, w=None, y=None, fit_intercept=True):
        """
        Check inputs for matching dimensions and convert to numpy arrays

        Add a column of ones to X if fit_intercept
        Assure y consists of zeros and ones

        :param X: feature matrix
        :param w: weight vector
        :param y: target vector
        :param fit_intercept:
        :return: X, w, y as numpy arrays
        """
        if y is not None:
            X = np.atleast_2d(X)
            y = (np.atleast_1d(y) != 0).astype(int)
            w = np.zeros(1, dtype=float) if w is None else np.atleast_1d(w)

            if X.ndim > 2:
                raise ValueError("Dimension of features X bigger than 2 not supported")
            if y.ndim > 1:
                raise ValueError("Dimension of target y bigger than 1 not supported")
            if w.ndim > 1:
                raise ValueError("Dimension of weights w bigger than 1 not supported")

            if X.shape[0] == 1 and X.shape[1] == y.shape[0]:
                # X could have been 1D and stored as a row
                # in stead of a column. If so transpose X.
                X = X.T
            elif X.shape[0] != y.shape[0]:
                raise ValueError("Number of data points in features X and target y don't match")

            if fit_intercept == True:
                X = np.concatenate((X, np.ones((X.shape[0], 1), dtype=float)), axis=1)

            if w.shape[0] == 1:
                # w could have been given as a number not an array
                # if so expand it to the width of X
                w = np.full(X.shape[1], w[0], dtype=float)

            if w.shape[0] != X.shape[1]:
                raise ValueError("Dimension of weights w does not match number of features X")

        else:
            X = np.atleast_2d(X)
            if fit_intercept == True:
                X = np.concatenate((X, np.ones((X.shape[0], 1), dtype=float)), axis=1)

            w = np.zeros(1, dtype=float) if w is None else np.atleast_1d(w)
            if w.shape[0] == 1:
                # w could have been given as a number not an array
                # if so expand it to the width of X
                w = np.full(X.shape[1], w[0], dtype=float)

            if X.ndim > 2:
                raise ValueError("Dimension of features X bigger than 2 not supported")
            if w.ndim > 1:
                raise ValueError("Dimension of weights w bigger than 1 not supported")

            if w.shape[0] != X.shape[1]:
                raise ValueError("Dimension of weights w does not match number of features X")

        return X, w, y
    
    
    
    
