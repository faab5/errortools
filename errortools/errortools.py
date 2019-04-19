import numpy as np
import iminuit

def estimate_errors_sampling(fnc, X, p, cvr_mtx, n_samples='auto', return_covariance=False, *args, **kwargs):
    """
    Estimate uncertainties via sampling the posterior

    :param fnc: function that takes features X and parameters p as arguments
        *args and **kwargs are passed onto this function
    :param X: [numpy.ndarray shape (n_data, n_features)] input features
    :param p: [numpy.ndarray shape (n_pars,)] parameters
    :param cvr_mtx: [numpy.ndarray shape (n_pars, n_pars)] covariance matrix of parameters
    :param n_samples: [int] sample size
        'auto' for automatically determining sample size
    :param return_covariance: [boolean] Whether to return the covariance matrix of the
        function value errors in stead of the error estimates themselves
    :param args: extra arguments for function
    :param kwargs: extra keyword arguments for function
    :return: [numpy.ndarray shape (n_data,)] uncertainty estimates of function values, or if
        return_covariance is True [numpy.ndarray shape (n_data, n_data,)] covariance matrix
        of errors
    """
    if n_samples == 'auto':
        # ToDo: Get a better based number of samples
        n_pars = p.shape[0]
        n_samples = 1000 if n_pars < 10 else 10000 if n_pars < 100 else 100000 if n_pars < 1000 else 100*n_pars

    sampled_parameters = np.random.multivariate_normal(p, cvr_mtx, n_samples).T  # shape (n_pars, n_samples, )
    sampled_function = np.array([fnc(X, q, *args, **kwargs) for q in sampled_parameters]).T # shape (n_data, n_samples, )
    reference_function = np.tile(fnc(X, p, *args, **kwargs), (n_samples, 1)).T # shape (n_data, n_samples,)

    function_variation = sampled_function - reference_function  # shape (n_data, n_samples,)

    if return_covariance == True:
        covar = np.dot(function_variation, function_variation.T) / n_samples  # shape (n_data, n_data,)
        return covar
    else:
        error = np.sqrt(np.mean(np.square(function_variation), axis=1))  # shape (ndata,)
        return error

def estimate_errors_linear(grad, cvr_mtx, return_covariance=False):
    """
    Estimate uncertainties via linear error propagation

    :param grad: [numpy.ndarray shape (n_data, n_pars,)] Gradients wrt parameters of a function
    :param cvr_mtx: [numpy.ndarray shape (n_pars, n_pars,)] Covariance matrix of the parameters
    :param return_covariance: [boolean] Whether to return the covariance matrix of the
        function value errors in stead of the error estimates themselves
    :return: [numpy.ndarray shape (n_data,)] uncertainty estimates of function values, or if
        return_covariance is True [numpy.ndarray shape (n_data, n_data,)] covariance matrix
        of errors
    """
    grad = np.atleast_2d(grad)
    cvr_mtx = np.atleast_2d(cvr_mtx)
    if cvr_mtx.ndim > 2:
        raise NotImplementedError("Multidimensional cases not implemented")
    if cvr_mtx.shape[0] != cvr_mtx.shape[1]:
        raise ValueError("Covariance matrix not square")
    if cvr_mtx.shape[0] != grad.shape[1]:
        raise ValueError("Shape mismatch between gradients and covariance matrix")

    if return_covariance == True:
        covar = np.dot(grad, np.dot(cvr_mtx, grad.T))
        return covar
    else:
        error = np.sqrt(np.abs([np.dot(g, np.dot(cvr_mtx, g)) for g in grad]))
        return error