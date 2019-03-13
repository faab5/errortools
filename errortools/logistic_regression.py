import numpy as np
import iminuit
import scipy.stats

class LogisticRegression():
    """

    """

    def __init__(self, fit_intercept=True, l1=0, l2=0):
        # pre-fit attributes
        self.fit_intercept = fit_intercept
        self.l1 = l1
        self.l2 = l2

        # post-fit attributes
        self.weights = None
        self.cvr_mtx = None
        self.minuit  = None

        # Fit parameters that a user could set
        # but probably won't, so good defaults
        self._minuit_limit = None
        self._minuit_fix = None
        self._minuit_name = None
        self._minuit_throw_nan = False
        self._minuit_pedantic = True
        self._minuit_print_level = 0
        self._migrad_ncall = 10000
        self._migrad_nsplit = 1
        self._migrad_precision = None
        self._hesse_maxcall = 0

    def negativeLogPosterior(self, w, X, y, l1, l2):
        """
        Calculates the negative of the log of the posterior
        distribution over the weights given targets and
        features.

        A combination of the negative log likelihood for
        classification (i.e. the log-loss), and l1 and/or l2
        regularization

        :param w:
        :param X:
        :param y:
        :param l1:
        :param l2:
        :return: negative log posterior of weights given data
        """
        # predictions on train set with given weights
        y_pred = scipy.stats.logistic.cdf(np.dot(X, w))

        # negative log-likelihood of predictions
        nll = -np.sum(np.where(y==1, np.log(y_pred), np.log(1-y_pred)))

        if l1 == 0 and l2 == 0:
            return nll

        # negative log-prior of weights
        nlp = np.mean(np.abs(l1 * w)) + np.mean(l2 * w**2)

        return nll + nlp

    def fit(self, X, y, w0=0, fit_intercept=None, l1=None, l2=None):
        """
        Fit logistic regression to feature matrix X and target vector y

        :param X: feature matrix
        :param y: target vector
        :param w0: initial weight vector
        :param fit_intercept: whether to include the intercept
            default None, taken as previously
        :param l1: l1 reguralization parameter
            default None, taken as previously
        :param l2: l2 regularization parameter
            default None, taken as previously
        """
        # update fit_intercept parameters if given
        if fit_intercept in (True, False,):
            self.fit_intercept = fit_intercept

        # check inputs
        X, w0, y = self._check_inputs(X, w0, y, self.fit_intercept)

        # update regularization parameters if given
        if isinstance(l1, (int, float,)):
            self.l1 = l1
        elif hasattr(l1, "__iter__"):
            l1 = np.array(l1, dtype=float)
            if l1.ndim == 1 and l1.shape[0] == w0.shape[0]:
                self.l1 = l1
        if isinstance(l2, (int, float,)):
            self.l2 = l2
        elif hasattr(l1, "__iter__"):
            l2 = np.array(l2, dtype=float)
            if l2.ndim == 1 and l2.shape[0] == w0.shape[0]:
                self.l2 = l2

        # define function to be minimized
        fcn = lambda w: self.negativeLogPosterior(w, X, y, self.l1, self.l2)

        # initiate minuit minimizer
        self.minuit = iminuit.Minuit.from_array_func(fcn=fcn, start=w0,
                throw_nan=self._minuit_throw_nan, pedantic=self._minuit_pedantic,
                print_level=self._minuit_print_level, grad=None, error=0,
                limit=self._minuit_limit, fix=self._minuit_fix, name=self._minuit_name,
                use_array_call=True, errordef=1)

        # minimize with migrad
        fmin, _ = self.minuit.migrad(ncall=self._migrad_ncall, nsplit=self._migrad_nsplit,
                precision=self._migrad_precision)

        # check validity of minimum
        if not fmin.is_valid:
            # ToDo check for more fitting failures
            raise RuntimeError("Minimization has not converged.\n%s" % (str(fmin)))

        # estimate covariance matrix with hesse
        self.minuit.hesse(maxcall=self._hesse_maxcall)

        # remember weights and covariance matrix
        self.weights = self.minuit.np_values()
        self.cvr_mtx = self.minuit.np_matrix()
    
    def predict(self, X):
        pass
      
    def predict(self, X, w):
        pass
    
    def estimate_errors(self, X):
        pass

    def _check_inputs(self, X, w=None, y=None, fit_intercept=True):
        """
        Check inputs for matching dimensions and convert to numpy arrays

        Adds a column of ones to X if fit_intercept
        Assures y consists of zeros and ones

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
    
    
    
    
