import numpy as np
import iminuit
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import scipy

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
    
def report_loss_versus_approximation(model, X, y, l1, l2, features, pdf, pdf_name = "report.pdf"):
    """
    Create a PDF report with plots showing the loss versus the parabolic approximation of the loss. 

    :param model: fitted model
    :param X: [numpy.ndarray shape (n_data, n_features)] input features
    :param y: targets for fitting
    :param l1: L1-regularization parameter. Multiplies the sum of absolute parameters
    :param l2: L2-regularization parameter. Multiplies half the sum of squared parameters
    :param features: list of input feature names
    :param pdf: PDF pages object
    :param pdf_name: name of the PDF document
    """
 
    # TODO scale figure as such that it has the same shape as previous pages
    # TODO check that the model provided is a fitted model
    fig, ax = plt.subplots(round(len(features)/2), 2, figsize=(20, 10))
    X_bias = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    f0 = model.negativeLogPosterior(model.parameters, X_bias, y, l1, l2)

    for p in range(0, model.parameters.shape[0]):  
        param_minimum = model.minuit.get_param_states()[p]['value']
        weights = np.linspace(param_minimum - 1, param_minimum + 1, 100)
        
        params = model.parameters.copy()
        loss = []
        approx = []

        for w in weights:
            params[p] = w
            loss.append(model.negativeLogPosterior(params, X_bias, y, l1, l2))
            parabolic_approx = params - model.parameters

            approx.append(f0 + 0.5 * np.array([np.dot(parabolic_approx, np.dot(scipy.linalg.inv(model.cvr_mtx), 
                                                                          parabolic_approx))]))

        col_ind = p % 2  
        row_ind = p // 2
        
        ax[row_ind ][col_ind].plot(weights, loss, '--', color='red', alpha=0.5, label="original")
        ax[row_ind][col_ind].plot(weights, approx, '-', color='orange', alpha=0.5, label="parabolic approximation")
        ax[row_ind][col_ind].set_xlabel(features[p])
        ax[row_ind][col_ind].set_title("logloss")
        ax[row_ind][col_ind].grid()
        ax[row_ind][col_ind].legend()

    if pdf == None:
        pdf = PdfPages(pdf_name)

    pdf.savefig(fig)
    
    return pdf

def report_parameter_error(model, pdf, features, pdf_name = "report.pdf"):
    """
    Create a PDF report showing the estimated error per parameter. 

    :param model: fitted model
    :param pdf: PDF pages object
    :param features: list of input feature names
    :param pdf_name: name of the PDF document
    """
    
    # TODO scale figure as such that it has the same shape as previous pages
    # TODO check that the model provided is a fitted model
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    
    ax.errorbar(x=np.arange(model.parameters.shape[0]), y=model.parameters, 
                yerr=np.sqrt(np.diag(model.cvr_mtx)), fmt='o', color='red',  alpha=0.6, markersize=10, 
                barsabove=True, capsize=10, label='fitted parameter value')
    ax.grid()
    ax.xaxis.set_ticks(np.arange(model.parameters.shape[0]))
    ax.xaxis.set_ticklabels(features + ['bias'])
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Fitted parameter value")
    
    if pdf == None:
        pdf = PdfPages(pdf_name)

    pdf.savefig(fig)
    
    return pdf

def report_correlation_matrix(model, pdf, features, pdf_name = "report.pdf"):
    """
    Create a PDF report showing the estimated error per parameter. 

    :param model: fitted model
    :param pdf: PDF pages object
    :param features: list of input feature names
    :param pdf_name: name of the PDF document
    """
    
    # TODO scale figure as such that it has the same shape as previous pages
    # TODO check that the model provided is a fitted model
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.axis('off')

    corr_matrix = model.minuit.np_matrix(correlation=True)
    
    ax.table(cellText=corr_matrix, rowLabels=features, colLabels=features, loc='center')
    ax.set_title("Correlation matrix")
    
    if pdf == None:
        pdf = PdfPages(pdf_name)
    
    pdf.savefig(fig)
    
    return pdf