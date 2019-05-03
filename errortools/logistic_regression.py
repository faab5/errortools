import numpy as np
import iminuit
import scipy.stats

class LogisticRegression(object):
    """
    Class for fitting, predicting and estimating error on logistic regression
    Quick usage:
    - instantiate:  m = LogisticRegression()
    - fit:          m.fit(X, y)
    - predict:      m.predict(X)
    - errors:       dwn, up = m.prediction_errors(X)

    Attributes:
    :param l1: L1-regularization parameter. Multiplies the sum of absolute parameters
    :param l2: L2-regularization parameter. Multiplies half the sum of squared parameters
    :param X: input features used to fit on
    :param y: targets used to fit on

    ToDo:
        Remove assertions
    """
    def __init__(self, l1=0, l2=0):
        """
        Instantiate a logistic regression
        :param l1: L1-regularization parameter. Multiplies the sum of absolute parameters
        :param l2: L2-regularization parameter. Multiplies half the sum of squared parameters
        """
        # pre-fit attributes
        self.l1 = l1
        self.l2 = l2

        # post-fit attributes
        self.minuit  = None

        # training data
        self.X = None
        self.y = None

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
        return np.sqrt(np.diag(self.minuit.np_covariance()))

    @property
    def cvr_mtx(self):
        """
        Covariance matrix of the fit parameters
        """
        if self.minuit is None:
            raise RuntimeError("Fit before access to fit parameters")
        return self.minuit.np_covariance()

    def negative_log_posterior(self, p, X, y):
        """
        Calculate the negative of the log of the posterior
        distribution over parameters given targets and
        features.
        A combination of the negative log likelihood for
        classification (i.e. the log-loss), and l1 and/or l2
        regularization
        :param p: [numpy 1D array] parameter vector
        :param X: [numpy 2D array] feature matrix
        :param y: [numpy 1D array] target vector
        :return: negative log posterior of parameters given data
        """

        assert p.shape[0] == X.shape[1] + 1

        # predictions on train set with given parameters
        y_pred = scipy.stats.logistic.cdf(X.dot(p[:-1]) + p[-1])

        # negative log-likelihood of predictions
        nll = -np.sum(y*np.log(y_pred+1e-16) + (1-y)*np.log(1-y_pred+1e-16))

        if self.l1 == 0 and self.l2 == 0:
            return nll

        # negative log-prior of parameters
        nlp = np.sum(np.abs(self.l1 * p)) + 0.5 * np.sum(self.l2 * p**2)

        return nll + nlp

    # alias
    loss = negative_log_posterior

    def gradient_negative_log_posterior(self, p, X, y):
        """
        Calculate the gradient of the negative of the
        log of the posterior distribution over parameters
        given targets and features

        :param p: [numpy 1D array] parameter vector
        :param X: [numpy 2D array] feature matrix
        :param y: [numpy 1D array] target vector
        :return: gradient with respect to the parameters of the negative
            log posterior
        """

        assert p.shape[0] == X.shape[1] + 1

        # predictions on train set with given parameters
        y_pred = scipy.stats.logistic.cdf(X.dot(p[:-1]) + p[-1])

        # gradient negative log-likelihood
        gnll = np.empty_like(p)
        gnll[:-1] = np.sum((y_pred-y)[:,np.newaxis] * X, axis=0)
        gnll[-1] = np.sum(y_pred-y)

        if self.l1 == 0 and self.l2 == 0:
            return gnll

        # gradient of negative log-prior
        gnlp = self.l1 * np.sign(p) + self.l2 * p

        return gnll + gnlp

    # alias
    grad_loss = gradient_negative_log_posterior

    def fit(self, X, y,
            initial_parameters=None, initial_step_sizes=None,
            parameter_limits=None,
            #parameter_fixes=None,
            print_level=0,
            #fit_intercept=None,
            max_function_calls=10000, n_splits=1):
        """
        Fit logistic regression to feature matrix X and target vector y
        
        If you call this method more than once, you resume a fit with
        parameters, step sizes, limits and fixes as they were at the end
        of the previous fit, for each that is given as None as an argument

        :param X: [numpy.ndarray shape (n_data, n_features,)] feature matrix
        :param y: [numpy.ndarray shape (n_data,)] target vector
        :param initial_parameters: [sequence of numbers, length n_features+1] initial parameter vector
            A single number is promoted to all parameters
            None means all zeros for a first fit, or resume from previous fit
        :param initial_step_sizes: [sequence of numbers, length n_features+1] initial minimization
             parameter step sizes. A single number is promoted to all parameters
             Usually, the choice is not important. In the worst case, iminuit will use a few more
             function evaluations to find the minimum
             None means all ones for a first fit, or resume from previous fit
        :param parameter_limits: [sequence of tuples of numbers, length n_features+1] lower and upper bounds
            for parameters. Use None within the sequence for no bound for that parameter or False for no bounds
            for all parameters, and use None to take the limits from the previous fit
        :param parameter_fixes: [sequence of booleans, length n_features+1] Whether to fix a parameter to the
            initial value
            Use False not to fix any parameters and None to take the fixes from the previous fit
        :param print_level: 0 is quiet. 1 print out fit results. 2 paranoid. 3 really paranoid
        :param fit_intercept: If given, overrides the last element of parameter_fixes
        :param max_function_calls: [integer] maximum number of function calls
        :param n_splits: [integer] split fit in to n_splits runs. Fitting stops when it found the function
            minimum to be valid or n_calls is reached
        """
        self.X, self.y = self._check_inputs(X, y)

        assert self.minuit is None or self.X.shape[1] + 1 == self.parameters.shape[0]

        if self.minuit is None or\
            initial_parameters is not None or\
            initial_step_sizes is not None or\
            parameter_limits is not None or\
            parameter_fixes is not None:

            n_dim = self.X.shape[1] + 1

            if initial_parameters is None:
                if self.minuit is not None:
                    initial_parameters = self.parameters
                else:
                    initial_parameters = np.zeros(n_dim, dtype=float)
            elif isinstance(initial_parameters, (float, int,)):
                initial_parameters = np.full(n_dim, initial_parameters, dtype=float)
            elif hasattr(initial_parameters, "__iter__"):
                initial_parameters = np.array(initial_parameters, dtype=float)
                if initial_parameters.shape[0] != n_dim:
                    raise ValueError("Dimensions of initial parameters don't match known dimensions")
            else:
                raise ValueError("Initial parameters not understood")

            if initial_step_sizes is not None:
                if isinstance(initial_step_sizes, (float, int,)):
                    initial_step_sizes = np.full(n_dim, initial_step_sizes, dtype=float)
                elif not hasattr(initial_step_sizes, "__iter__") or isinstance(initial_step_sizes, str):
                    raise ValueError("Step sizes should be a sequence of numbers")
                elif not all([isinstance(s, (float, int,)) for s in initial_step_sizes]):
                    raise ValueError("Step sizes should be a sequence of numbers")
                elif len(initial_step_sizes) != n_dim:
                    raise ValueError("{:d} step sizes given for {:d} parameters".format(len(initial_step_sizes), n_dim))
            elif self.minuit is not None:
                initial_step_sizes = [state['error'] for state in self.minuit.get_param_states()]
            else:
                initial_step_sizes = 1

            if parameter_limits == False:
                parameter_limits = None
            elif parameter_limits is not None:
                if not hasattr(parameter_limits, "__iter__") or isinstance(parameter_limits, str):
                    raise ValueError("Limits should be a sequence of range tuples")
                if not all([l is None or (isinstance(l,(tuple,)) and len(l)==2 and\
                        (l[0] is None or isinstance(l[0],(int,float,))) and\
                        (l[1] is None or isinstance(l[1],(int,float,)))) for l in parameter_limits]):
                    raise ValueError("A limit should be a range tuple or None")
                if len(parameter_limits) != n_dim:
                    raise ValueError("{:d} limits given for {:d} parameters".format(len(parameter_limits), n_dim))
            elif self.minuit is not None:
                parameter_limits = [(state['lower_limit'], state['upper_limit'],) for state in self.minuit.get_param_states()]

            if parameter_fixes == False:
                parameter_fixes = None
            elif parameter_fixes is not None:
                if not hasattr(parameter_fixes, "__iter__") or isinstance(parameter_fixes, str):
                    raise ValueError("Fixes should be a sequence of booleans")
                if not all([isinstance(f, (bool, int, float,)) for f in parameter_fixes]):
                    raise ValueError("A fix should be True or False")
                if len(parameter_fixes) != n_dim:
                    raise ValueError("{:d} fixes given for {:d} parameters".format(len(parameter_fixes), n_dim))
                parameter_fixes = [bool(f) for f in parameter_fixes]
            elif self.minuit is not None:
                parameter_fixes = [state['is_fixed'] for state in self.minuit.get_param_states()]

            if fit_intercept in (True, False):
                parameter_fixes[-1] = fit_intercept

            # define function to be minimized
            fcn = lambda p: self.negative_log_posterior(p, self.X, self.y)

            # define the gradient of the function to be minimized
            grd = lambda p: self.gradient_negative_log_posterior(p, self.X, self.y)

            # initiate minuit minimizer
            self.minuit = iminuit.Minuit.from_array_func(fcn=fcn,
                start=initial_parameters, error=initial_step_sizes,
                limit=parameter_limits, fix=parameter_fixes,
                throw_nan=True, print_level=print_level,
                grad=grd, use_array_call=True, errordef=0.5, pedantic=False)

        self.minuit.print_level = print_level

        # minimize with migrad
        fmin, _ = self.minuit.migrad(ncall=max_function_calls, nsplit=n_splits, resume=True)

        # check validity of minimum
        if not fmin.is_valid:
            if not fmin.has_covariance or not fmin.has_accurate_covar or not fmin.has_posdef_covar or \
                    fmin.has_made_posdef_covar or fmin.hesse_failed:
                # It is known that migrad sometimes fails calculating the covariance matrix,
                # but succeeds on a second try
                self.minuit.set_strategy(2)
                fmin, _ = self.minuit.migrad(ncall=max_function_calls, nsplit=n_splits, resume=True)
                if not fmin.is_valid:
                    raise RuntimeError("Problem encountered with minimization.\n%s" % (str(fmin)))

        self.minuit.hesse()

    def predict(self, X):
        """
        Calculate the logistic scores given features X

        :param X: [numpy 2D array] feature matrix
        :return: [numpy 1D array] logistic regression scores
        """
        X, _ = self._check_inputs(X, None)
        p = self.parameters
        y_pred = scipy.stats.logistic.cdf(X.dot(p[:-1]) + p[-1])
        return y_pred

    def prediction_errors(self, X, method="interval", **kwargs):
        """
        Estimate asymmetric errors on predictions

        All methods rely on a multivariate Gaussian approximation of the likelihood around the maximum
        - The interval method finds the minimum and maximum prediction values, when parameters are taken
            from within uncertainty ellipses around the maximum likely parameters
        - The sampling method samples the full mv-Gaussian to calculate the variance of the prediction
        - The linear method approximates the prediction linearly around the maximum parameters,
            so that the variance of the prediction is simple vector arithmetic

        :param X:
        :param method: [str] method to use
            One out of interval (default), asmpling, or linear_error_propagation
            See methods for details
        :param kwargs: keyword arguments passed onto method
            Possibilities are n_samples, and n_stddevs
            Defaults apply if empty
        :return: [(numpy 1D array, numpy 1D array,)] lower and upper error estimates
        """
        if method == "sampling":
            kwargs =  {key: kwargs[key] for key in ('n_samples',) if key in kwargs.keys()}
            lower = upper = self.prediction_errors_from_sampling(X, return_covariance=False, **kwargs)
        elif method == "linear_error_propagation" or method == "linear":
            kwargs =  {key: kwargs[key] for key in ('n_stddevs',) if key in kwargs.keys()}
            lower = upper = self.prediction_errors_from_linear_error_propagation(X, return_covariance=False, **kwargs)
        else:
            kwargs = {key: kwargs[key] for key in ('n_stddevs',) if key in kwargs.keys()}
            lower, upper =  self.prediction_errors_from_interval(X, **kwargs)
        return lower, upper

    def prediction_errors_from_interval(self, X, n_stddevs=1):
        """
        Estimate upper and lower uncertainties

        This method is based on error intervals, where every standard
        deviation interval in parameter space is the multi-dimensional
        range where the negative log-likelihood goes up by 1/2
        The lower and upper errors are the maximum and minimum amount
        respectively that the logistic function goes down or up when
        taking parameters within this interval
        :param X: [numpy 2D array] feature matrix
        :param nstddevs: [int] error contour
        :return: [numpy 1D arrays] lower and upper error estimates
        """
        X, _ = self._check_inputs(X, None)
        X_biased = np.concatenate((X, np.ones((X.shape[0], 1), dtype=float)), axis=1)
        p = self.parameters
        mid = X_biased.dot(p)
        delta = np.array([np.sqrt(np.abs(np.dot(u,np.dot(self.cvr_mtx, u)))) for u in X_biased], dtype=float)
        y_pred = scipy.stats.logistic.cdf(mid)
        upper = scipy.stats.logistic.cdf(mid + n_stddevs * delta) - y_pred
        lower = y_pred - scipy.stats.logistic.cdf(mid - n_stddevs * delta)
        return lower, upper
    
    def prediction_errors_from_sampling(self, X, n_samples=10000, return_covariance=False):
        """
        Estimate uncertainties via sampling the posterior

        Calculates the non-central variance for each input data point
        by sampling parameters from an approximate posterior (a multivariate
        normal distribution)

        :param X: [numpy 2D array] feature matrix
        :param n_samples: [int] number of samples to draw
        :param return_covariance: [boolean] return only error estitimes (False),
            or full covariance matrix (True) of the estimates
        :return: covariance matrix of error estimates if return_covariance
            is True, otherwise the upper and lower error estimates (symmetric)
        """
        X, _ = self._check_inputs(X, None)

        if not isinstance(n_samples, (int,)):
            raise ValueError("Non-integer number of samples provided")

        X_biased = np.concatenate((X, np.ones((X.shape[0], 1), dtype=float)), axis=1)

        p = self.parameters

        sampled_parameters = np.random.multivariate_normal(p, self.cvr_mtx, n_samples).T # shape (npars, nsamples,)
        fitted_parameters = np.tile(p, (n_samples, 1)).T # shape (npars, nsamples,)

        sigmoid_sampled_parameters = scipy.stats.logistic.cdf(X_biased.dot(sampled_parameters)) # shape (ndata, nsamples,)
        sigmoid_fitted_parameters = scipy.stats.logistic.cdf(X_biased.dot(fitted_parameters)) # shape (ndata, nsamples,)
        sigmoid_variation = sigmoid_sampled_parameters - sigmoid_fitted_parameters  # shape (ndata, nsamples,)

        if return_covariance == True:
            covar = np.dot(sigmoid_variation, sigmoid_variation.T) / n_samples # shape (ndata, ndata,)
            return covar
        else:
            var = np.mean(np.square(sigmoid_variation), axis = 1) # shape (ndata,)
            symmetric_error = np.sqrt(np.abs(var))
            return symmetric_error

    def prediction_errors_from_linear_error_propagation(self, X, n_stddevs=1, return_covariance=False):
        """
        Estimate uncertainties via linear error propagation
        
        Calculates the non-central variance for each input data point
        by approximating the logistic function linearly.
        This method is fast, but may be inaccurate

        :param X: [numpy 2D array] feature matrix
        :param n_stddevs: [int] number of standard deviations to estimate gradient on
            None means take exact gradient
        :return: covariance matrix of error estimates if return_covariance
            is True, otherwise the upper and lower error estimates (symmetric)
        """
        X, _ = self._check_inputs(X, None)

        X_biased = np.concatenate((X, np.ones((X.shape[0], 1), dtype=float)), axis=1)

        fcn = lambda p: scipy.stats.logistic.cdf(X_biased.dot(p))

        p = self.parameters

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
            return symmetric_error

    def _check_inputs(self, X, y=None):
        """
        Check inputs for matching dimensions and convert to numpy arrays
        
        :param X: feature matrix
        :param y: target vector
        :return: X, y as numpy arrays
        """
        X = np.array(X)

        if X.ndim > 2:
            raise ValueError("Dimension of features X bigger than 2 not supported")
        elif X.ndim == 1:
            X = X[:, np.newaxis]

        if self.minuit is not None:
            p = self.minuit.np_values()

            if X.shape[1] + 1 == p.shape[0]:
                pass
            elif X.shape[1] == 1 and X.shape[0] + 1 == p.shape[0]:
                X = X.T
            else:
                raise ValueError("Dimension of X do not match dimensions of parameters")

        if y is not None:
            y = (np.atleast_1d(y) != 0).astype(int)

            if y.ndim > 1:
                raise ValueError("Dimension of target y bigger than 1 not supported")

            if X.shape[0] != y.shape[0]:
                raise ValueError("Number of data points in features X and target y don't match")

        return X, y
