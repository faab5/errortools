import numpy as np
import iminuit
import scipy.stats
import scipy.sparse
import matplotlib.pyplot as plt

class LogisticRegression(object):
    """
    Class for fitting a, predicting with and estimating error on logistic regression


    """
    def __init__(self, fit_intercept=True, l1=0, l2=0):
        # pre-fit attributes
        self.fit_intercept = fit_intercept
        self.l1 = l1
        self.l2 = l2

        # post-fit attributes
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

    @property
    def weights(self):
        if self.minuit is None:
            raise RuntimeError("Fit before access to fit parameters")
        return self.minuit.np_values()

    @property
    def cvr_mtx(self):
        if self.minuit is None:
            raise RuntimeError("Fit before access to fit parameters")
        return self.minuit.np_matrix()

    @staticmethod
    def negativeLogPosterior(w, X, y, l1, l2):
        """
        Calculates the negative of the log of the posterior
        distribution over the weights given targets and
        features.

        A combination of the negative log likelihood for
        classification (i.e. the log-loss), and l1 and/or l2
        regularization

        :param w: [numpy 1D array] weight vector
        :param X: [numpy 2D array] feature matrix
        :param y: [numpy 1D array] target vector
        :param l1: [float or numpy 1D array] l1 regularization parameter
        :param l2: [float or numpy 2D array] l2 regularization parameter
        :return: negative log posterior of weights given data
        """
        # predictions on train set with given weights
        y_pred = 1./(1.+np.exp(-X.dot(w)))

        # negative log-likelihood of predictions
        nll = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

        if l1 == 0 and l2 == 0:
            return nll

        # negative log-prior of weights
        nlp = np.sum(np.abs(l1 * w)) + 0.5 * np.sum(l2 * w**2)

        return nll + nlp

    @staticmethod
    def gradientNegativeLogPosterior(w, X, y, l1, l2):
        """

        :param w: [numpy 1D array] weight vector
        :param X: [numpy 2D array] feature matrix
        :param y: [numpy 1D array] target vector
        :param l1: [float or numpy 1D array] l1 regularization parameter
        :param l2: [float or numpy 2D array] l2 regularization parameter
        :return: gradient with respect to the weights of the negative
            log posterior
        """
        # predictions on train set with given weights
        y_pred = 1. / (1. + np.exp(-X.dot(w)))

        # gradient negative log-likelihood
        gnll = np.sum((y_pred-y)[:,np.newaxis] * X, axis=0)

        if l1 == 0 and l2 == 0:
            return gnll

        # gradient of negative log-prior
        gnlp = l1 * np.sign(w) + l2 * w

        return gnll + gnlp

    def fit(self, X, y, w0=0, fit_intercept=None, l1=None, l2=None):
        """
        Fit logistic regression to feature matrix X and target vector y

        :param X: feature matrix
        :param y: target vector
        :param w0: initial weight vector
        :param fit_intercept: override whether to include the intercept
            default None, taken as previously
        :param l1: override l1 reguralization parameter
            default None, taken as previously set
        :param l2: override l2 regularization parameter
            default None, taken as previously set
        """
        # update fit_intercept parameters if given
        if fit_intercept in (True, False,):
            self.fit_intercept = fit_intercept

        # check inputs
        X, y, w0 = self._check_inputs(X, y, w0, self.fit_intercept)

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
            raise RuntimeError("Minimization has not converged.\n%s" % (str(fmin)))
        if not fmin.has_covariance or not fmin.has_accurate_covar or not fmin.has_posdef_covar or \
                fmin.has_made_posdef_covar or fmin.hesse_failed:
            raise RuntimeError("Problem encountered with covariance estimation.\n%s" % (str(fmin)))

        self.minuit.hesse(maxcall=self._hesse_maxcall)

    def predict(self, X):
        """
        Calculates logistic scores given features X

        :param X: [numpy 2D array] feature matrix
        :return: [numpy 1D array] logistic regression scores
        """
        X, _, w = self._check_inputs(X, None, self.weights, self.fit_intercept)
        y_pred = 1. / (1. + np.exp(-X.dot(w)))
        return y_pred
      
    def estimate_errors(self, X, nstddevs=1):
        """
        Calculates upper and lower uncertainty estimates
        on logistic scores for given features X

        The log-posterior distribution is approximated with a
        parabolic approximation (a covariance matrix
        around the fitted weights), i.e. as a Gaussian
        distribution.
        The one standard deviation interval in weight space is
        then an ellipsis around the fitted weight vector.
        Within this one standard deviation interval there is a
        weight vector for which the logistic score is maximum and
        there is a weight vector for which it is minimum.
        These maximum and minimum are then quoted as the one
        standard deviation upper and lower errors on the logistic
        score.

        :param X: [numpy 2D array] feature matrix
        :param nstddevs: [int] number of standard deviations away from
            fitted weights
        :return: [numpy 1D arrays] upper and lower error estimates
        """
        X, _, w = self._check_inputs(X, None, self.weights, self.fit_intercept)
        mid = X.dot(w)
        delta = np.array([np.sqrt(np.abs(np.dot(u,np.dot(self.cvr_mtx, u)))) for u in X], dtype=float)
        y_pred = 1. / (1. + np.exp(-mid))
        upper = 1. / (1. + np.exp(-mid - nstddevs * delta)) - y_pred
        lower = y_pred - 1. / (1. + np.exp(-mid + nstddevs * delta))
        return lower, upper

    def _check_inputs(self, X, y=None, w=None, fit_intercept=True):
        """
        Check inputs for matching dimensions and convert to numpy arrays

        Adds a column of ones to X if fit_intercept
        Assures y consists of zeros and ones

        :param X: feature matrix
        :param w: weight vector
        :param y: target vector
        :param fit_intercept: whether to fit the intercept
        :return: X, y, w, as numpy arrays
        """
        X = np.atleast_2d(X)

        w = np.zeros(1, dtype=float) if w is None else np.atleast_1d(w)

        if X.ndim > 2:
            raise ValueError("Dimension of features X bigger than 2 not supported")
        if w.ndim > 1:
            raise ValueError("Dimension of weights w bigger than 1 not supported")

        if y is not None:
            y = (np.atleast_1d(y) != 0).astype(int)

            if y.ndim > 1:
                raise ValueError("Dimension of target y bigger than 1 not supported")

            if X.shape[0] == 1 and X.shape[1] == y.shape[0]:
                # X could have been 1D and stored as a row
                # in stead of a column. If so transpose X.
                X = X.T
            elif X.shape[0] != y.shape[0]:
                raise ValueError("Number of data points in features X and target y don't match")

        if fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1), dtype=float)), axis=1)

        if w.shape[0] == 1:
            # w could have been given as a number not an array
            # if so expand it to the width of X
            w = np.full(X.shape[1], w[0], dtype=float)

        if w.shape[0] != X.shape[1]:
            raise ValueError("Dimension of weights w does not match number of features X")

        return X, y, w