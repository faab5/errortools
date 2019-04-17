import numpy as np
import iminuit
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

    @property
    def parameters(self):
        """
        Fit parameters, a.k.a. weights
        """
        if self.minuit is None:
            raise RuntimeError("Fit before access to fit parameters")
        return self.minuit.np_values()

    @property
    def errors(self):
        """
        Errors of the fit parameters
        Square root of the diagonal of the covariance matrix
        """
        if self.minuit is None:
            raise RuntimeError("Fit before access to fit parameters")
        return np.sqrt(np.diag(self.minuit.np_matrix()))

    @property
    def cvr_mtx(self):
        """
        Covariance matrix of the fit parameters
        """
        if self.minuit is None:
            raise RuntimeError("Fit before access to fit parameters")
        return self.minuit.np_matrix()

    @staticmethod
    def negativeLogPosterior(p, X, y, l1, l2):
        """
        Calculate the negative of the log of the posterior
        distribution over the parameters given targets and
        features.

        A combination of the negative log likelihood for
        classification (i.e. the log-loss), and l1 and/or l2
        regularization

        :param p: [numpy 1D array] parameter vector
        :param X: [numpy 2D array] feature matrix
        :param y: [numpy 1D array] target vector
        :param l1: [float or numpy 1D array] l1 regularization parameter
        :param l2: [float or numpy 2D array] l2 regularization parameter
        :return: negative log posterior of parameters given data
        """
        # predictions on train set with given parameters
        y_pred = 1./(1.+np.exp(-X.dot(p)))

        # negative log-likelihood of predictions
        nll = -np.sum(y*np.log(y_pred+1e-16) + (1-y)*np.log(1-y_pred+1e-16))

        if l1 == 0 and l2 == 0:
            return nll

        # negative log-prior of parameters
        nlp = np.sum(np.abs(l1 * p)) + 0.5 * np.sum(l2 * p**2)

        return nll + nlp

    @staticmethod
    def gradientNegativeLogPosterior(p, X, y, l1, l2):
        """

        :param p: [numpy 1D array] parameter vector
        :param X: [numpy 2D array] feature matrix
        :param y: [numpy 1D array] target vector
        :param l1: [float or numpy 1D array] l1 regularization parameter
        :param l2: [float or numpy 2D array] l2 regularization parameter
        :return: gradient with respect to the parameters of the negative
            log posterior
        """
        # predictions on train set with given parameters
        y_pred = 1. / (1. + np.exp(-X.dot(p)))

        # gradient negative log-likelihood
        gnll = np.sum((y_pred-y)[:,np.newaxis] * X, axis=0)

        if l1 == 0 and l2 == 0:
            return gnll

        # gradient of negative log-prior
        gnlp = l1 * np.sign(p) + l2 * p

        return gnll + gnlp

    def fit(self, X, y,
            initial_parameters=0, initial_step_sizes=1,
            parameter_limits=None, parameter_fixes=None,
            throw_nan=False, print_level=0,
            max_function_calls=10000, n_splits=1, precision=None):
        """
        Fit logistic regression to feature matrix X and target vector y

        :param X: [numpy.ndarray shape (n_data, n_features,)] feature matrix
        :param y: [numpy.ndarray shape (n_data,)]target vector
        :param initial_parameters: [sequence of numbers, length n_features+1] initial parameter vector
            A single number is promoted to all parameters
        :param initial_step_sizes: [sequence of numbers, length n_features+1] initial minimization
             parameter step sizes. A single number is promoted to all parameters
             Usually, the choice is not important. In the worst case, iminuit will use a few more
             function evaluations to find the minimum
        :param parameter_limits: [sequence of tuples of numbers, length n_features+1] lower and upper bounds
            for parameters. Use None for no bound
        :param parameter_fixes: [sequence of booleans, length n_features+1] Whether to fix a parameter to the
            initial value
        :param throw_nan: Minuit raises a RuntimeError when it encounters nan
        :param print_level: 0 is quiet. 1 print out fit results
        :param max_function_calls: [integer] maximum number of function calls
        :param n_splits: [integer] split fit in to n_splits runs. Fitting stops when it found the function
            minimum to be valid or n_calls is reached
        :param precision: override Miniut own’s internal precision
        """
        # check inputs
        X, y, initial_parameters = self._check_inputs(X, y, initial_parameters, self.fit_intercept)

        if parameter_limits is not None:
            if not hasattr(parameter_limits, "__iter__"):
                raise ValueError("Limits should be a sequence of range tuples")
            if not all([l is None or (isinstance(l,(tuple,)) and len(l)==2 and\
                    (l[0] is None or isinstance(l[0],(int,float,))) and\
                    (l[1] is None or isinstance(l[1],(int,float,)))) for l in parameter_limits]):
                raise ValueError("A limit should be a range tuple or None")
            if len(parameter_limits) != len(initial_parameters):
                raise ValueError("{:d} limits given for {:d} parameters".format(len(parameter_limits), len(initial_parameters)))

        if parameter_fixes is not None:
            if not hasattr(parameter_fixes, "__iter__"):
                raise ValueError("Fixes should be a sequence of booleans")
            if not all([isinstance(f, (bool, int, float,)) for f in parameter_fixes]):
                raise ValueError("A fix should be True or False")
            if len(parameter_fixes) != len(initial_parameters):
                raise ValueError("{:d} fixes given for {:d} parameters".format(len(parameter_fixes), len(initial_parameters)))
            fixes = [bool(f) for f in parameter_fixes]

        # define function to be minimized
        fcn = lambda p: self.negativeLogPosterior(p, X, y, self.l1, self.l2)

        # define the gradient of the function to be minimized
        grd = lambda p: self.gradientNegativeLogPosterior(p, X, y, self.l1, self.l2)

        # initiate minuit minimizer
        self.minuit = iminuit.Minuit.from_array_func(fcn=fcn, start=initial_parameters, error=initial_step_sizes,
                limit=parameter_limits, fix=parameter_fixes,
                throw_nan=throw_nan, print_level=print_level,
                grad=grd, use_array_call=True, errordef=0.5, pedantic=False)

        # minimize with migrad
        fmin, _ = self.minuit.migrad(ncall=max_function_calls, nsplit=n_splits, precision=precision)

        # check validity of minimum
        if not fmin.is_valid:
            if not fmin.has_covariance or not fmin.has_accurate_covar or not fmin.has_posdef_covar or \
                    fmin.has_made_posdef_covar or fmin.hesse_failed:
                # It is known that migrad sometimes fails calculating the covariance matrix,
                # but succeeds on a second try
                self.minuit.set_strategy(2)
                fmin, _ = self.minuit.migrad(ncall=max_function_calls, nsplit=n_splits, precision=precision, resume=True)
                if not fmin.is_valid:
                    raise RuntimeError("Problem encountered with minimization.\n%s" % (str(fmin)))

        self.minuit.hesse()

    def predict(self, X):
        """
        Calculates logistic scores given features X

        :param X: [numpy 2D array] feature matrix
        :return: [numpy 1D array] logistic regression scores
        """
        X, _, p = self._check_inputs(X, None, self.parameters, self.fit_intercept)
        y_pred = 1. / (1. + np.exp(-X.dot(p)))
        return y_pred
      
    def estimate_errors(self, X, nstddevs=1):
        """
        Calculate upper and lower uncertainty estimates
        on logistic scores for given features X, based on
        error contours/ellipses

        The log-posterior distribution is approximated with a
        parabolic approximation (a covariance matrix
        around the fitted parameters), i.e. as a Gaussian
        distribution.
        The one standard deviation interval in parameter space is
        then an ellipsis around the fitted parameter vector.
        On this one standard deviation interval there is a
        parameter vector for which the logistic score is maximum and
        there is a parameter vector for which it is minimum.
        These maximum and minimum are then quoted as the one
        standard deviation upper and lower errors on the logistic
        score.

        :param X: [numpy 2D array] feature matrix
        :param nstddevs: [int] error contour
        :return: [numpy 1D arrays] upper and lower error estimates
        """
        X, _, p = self._check_inputs(X, None, self.parameters, self.fit_intercept)
        mid = X.dot(p)
        delta = np.array([np.sqrt(np.abs(np.dot(u,np.dot(self.cvr_mtx, u)))) for u in X], dtype=float)
        y_pred = 1. / (1. + np.exp(-mid))
        upper = 1. / (1. + np.exp(-mid - nstddevs * delta)) - y_pred
        lower = y_pred - 1. / (1. + np.exp(-mid + nstddevs * delta))
        return lower, upper
    
    def estimate_errors_sampling(self, X, n_samples='auto', return_covariance=False):
        """
        Estimate uncertainties via linear sampling the posterior

        This is achieved by calculating the non-central variance 
        for each data point based on sampled parameters from a multivariate
        normal distribution and the parameters fitted by the logistic model.

        :param X: [numpy 2D array] feature matrix
        :param n_samples: [int] number of samples to draw from the distribution
            auto (default) automatically determines the number of samples
        :param return_covariance: [boolean] return only error estitimes (False),
            or full covariance matrix (True) of the estimates
        :return: covariance matrix of error estimates if return_covariance
            is True, otherwise the upper and lower error estimates (symmetric)
        """
        X, _, p = self._check_inputs(X, None, self.parameters, self.fit_intercept)

        if n_samples == 'auto':
            # ToDo: Get a better based number of samples
            ndim = self.parameters.shape[0]
            n_samples = 1000 if ndim < 10 else 10000 if ndim < 100 else 100000 if ndim < 10000 else ndim

        sampled_parameters = np.random.multivariate_normal(p, self.cvr_mtx, n_samples).T # shape (npars, nsamples,)
        fitted_parameters = np.tile(p, (n_samples, 1)).T # shape (npars, nsamples,)
        
        sigmoid_sampled_parameters = 1./(1.+np.exp(-X.dot(sampled_parameters))) # shape (ndata, nsamples,)
        sigmoid_fitted_parameters = 1./(1.+np.exp(-X.dot(fitted_parameters))) # shape (ndata, nsamples,)
        sigmoid_variation = sigmoid_sampled_parameters - sigmoid_fitted_parameters  # shape (ndata, nsamples,)

        if return_covariance == True:
            covar = np.dot(sigmoid_variation, sigmoid_variation.T) / n_samples # shape (ndata, ndata,)
            return covar
        else:
            var = np.mean(np.square(sigmoid_variation), axis = 1) # shape (ndata,)
            symmetric_error = np.sqrt(np.abs(var))
            return symmetric_error, symmetric_error

    def estimate_errors_linear(self, X, n_stddevs=1, return_covariance=False):
        """
        Estimate uncertainties via linear error propagation

        This is a good estimate of symmetric uncertainties if the function
        is monotonic in each dimension, which logistic function is.
        It is fast besides. It can only give symmetric error estimates though

        Approximates the posterior as a multivariate normal distribution with its covariance matrix C
        Approximates the logistic function linearly with its gradients g
        Uncertainties are then simply sqrt(g*C*g)

        :param X: [numpy 2D array] feature matrix
        :param n_stddevs: [int] number of standard deviations to estimate gradient on
            None means take exact gradient
        :return: covariance matrix of error estimates if return_covariance
            is True, otherwise the upper and lower error estimates (symmetric)
        """
        X, _, p = self._check_inputs(X, None, self.parameters, self.fit_intercept)

        fcn = lambda v: 1. / (1. + np.exp(-X.dot(v)))

        if isinstance(n_stddevs, (float, int,)):
            gradients = np.array([(fcn(p + n_stddevs*u) - fcn(p - n_stddevs*u)) \
                                  /(2 * np.sum(n_stddevs*u)) for u in np.diag(np.sqrt(np.diag(self.cvr_mtx)))]).T
        else:
            gradients = X * (fcn(p) * (1 - fcn(p)))[:, np.newaxis] # shape (ndata, npars,)

        if return_covariance == True:
            covar = np.dot(gradients, np.dot(self.cvr_mtx, gradients.T))  # shape (ndata, ndata,)
            return covar
        else:
            symmetric_error = np.sqrt(np.abs([np.dot(g, np.dot(self.cvr_mtx, g)) for g in gradients]))
            return symmetric_error, symmetric_error

    def _check_inputs(self, X, y=None, p=None, fit_intercept=True):
        """
        Check inputs for matching dimensions and convert to numpy arrays

        Adds a column of ones to X if fit_intercept
        Assures y consists of zeros and ones

        :param X: feature matrix
        :param p: parameter vector
        :param y: target vector
        :param fit_intercept: whether to fit the intercept
        :return: X, y, p, as numpy arrays
        """
        X = np.atleast_2d(X)

        p = np.zeros(1, dtype=float) if p is None else np.atleast_1d(p)

        if X.ndim > 2:
            raise ValueError("Dimension of features X bigger than 2 not supported")
        if p.ndim > 1:
            raise ValueError("Dimension of parameters p bigger than 1 not supported")

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

        if p.shape[0] == 1:
            # p could have been given as a number not an array
            # if so expand it to the width of X
            p = np.full(X.shape[1], p[0], dtype=float)

        if p.shape[0] != X.shape[1]:
            raise ValueError("Dimension of parameters does not match number of features X")

        return X, y, p
