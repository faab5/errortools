import numpy as np
import iminuit
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import scipy

def errors_from_sampling(fnc, X, p, cvr_mtx, n_samples='auto', return_covariance=False, *args, **kwargs):
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

def errors_from_linear_error_propagation(grad, cvr_mtx, return_covariance=False):
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
    
def report_loss_versus_approximation(model, X, y, features, pdf=None, pdf_name = "report.pdf"):
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

    #X_bias = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    #f0 = model.negative_log_posterior(model.parameters, X_bias, y)
    f0 = model.negative_log_posterior(model.parameters, X, y)
    
    if pdf == None:
        pdf = PdfPages(pdf_name)

    for p in range(0, model.parameters.shape[0]):  
        fig, ax = plt.subplots(1, 1, figsize=(8,4))

        param_minimum = model.minuit.get_param_states()[p]['value']
        weights = np.linspace(param_minimum - 1, param_minimum + 1, 100)
        
        params = model.parameters.copy()
        loss = []
        approx = []

        for w in weights:
            params[p] = w
            #loss.append(model.negative_log_posterior(params, X_bias, y))
            loss.append(model.negative_log_posterior(params, X, y))
            parabolic_approx = params - model.parameters

            approx.append(f0 + 0.5 * np.array([np.dot(parabolic_approx, np.dot(scipy.linalg.inv(model.cvr_mtx),
                                                                               parabolic_approx))]))

        col_ind = p % 2  
        row_ind = p // 2
        
        ax.plot(weights, loss, '--', color='red', alpha=0.5, label="original")
        ax.plot(weights, approx, '-', color='orange', alpha=0.5, label="parabolic approximation")
        ax.set_xlabel(features[p])
        ax.set_title("logloss")
        ax.grid()
        ax.legend()
        pdf.savefig(fig)
    
    return pdf


def report_parameter_error(model, features, pdf=None, pdf_name = "report.pdf",
                           figsize=(8, 4), rotation_x_labels = 20):
    """
    Create a PDF report showing the estimated error per parameter. 

    :param model: fitted model
    :param pdf: PDF pages object
    :param features: list of input feature names
    :param pdf_name: name of the PDF document
    :param figsize: size of the figure (tuple)
    :param rotation_x_labels: rotation of the labels on the x-axis (degrees or keyword)
    """
    
    # TODO scale figure as such that it has the same shape as previous pages
    # TODO check that the model provided is a fitted model

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.errorbar(x=np.arange(model.parameters.shape[0]), y=model.parameters, 
                yerr=np.sqrt(np.diag(model.cvr_mtx)), fmt='o', color='red',  alpha=0.6, markersize=10, 
                barsabove=True, capsize=10, label='fitted parameter value')
    ax.grid()
    ax.xaxis.set_ticks(np.arange(model.parameters.shape[0]))
    ax.xaxis.set_ticklabels(features + ['bias'], rotation=rotation_x_labels)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Fitted parameter value")
    
    if pdf == None:
        pdf = PdfPages(pdf_name)

    pdf.savefig(fig)
    
    return pdf

def report_correlation_matrix(model, features, pdf=None, pdf_name = "report.pdf"):
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

def expand(point, idx, rnge):
    """
    Expand a numpy array and replace the values at a specified index by a range of values
    
    :param point: the data point to be expanded
    :param idx: the index of the column to be replace by the values in rnge
    :param rnge: the values to replace in the data point
    """
    x = np.repeat(point, len(rnge)).reshape(len(point), len(rnge)).T
    x[:, idx] = rnge
    return x

def report_error_indivial_pred(model, sample, param, features, x_min, x_max, 
                               stepsize, pdf=None, pdf_name='report.pdf', figsize=(8,4)):
    """
    Create a PDF report showing the estimated error for an individual data sample by varying one dimension of the parameters. 

    :param model: fitted model
    :param sample: [numpy.ndarray shape (1, n_features)] data sample 
    :param param: the parameter that will be varied in the sample
    :param features: list of input feature names
    :param x_min: minimum value for the x-axis
    :param x_max: maximum value for the x-axis
    :param stepsize: the number of steps between x_min and x_max
    :param pdf: PDF pages object
    :param pdf_name: name of the PDF document
    """
    # TODO check that the model provided is a fitted model

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    param_index = features.index(param)

    x = np.linspace(x_min, x_max, stepsize)
    expanded_X = expand(sample, param_index, x)
    
    y_pred = model.predict(expanded_X)
    el, eu = model.prediction_errors_from_interval(expanded_X)
    ax.fill_between(x, y_pred-el, y_pred+eu, alpha=0.5, color='orange')
    ax.plot(x, y_pred, '-', color='orange')

    ax.set_ylim(0,1)
    ax.set_xlabel(param)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction error estimation")
    ax.grid()
    
    return pdf

def get_positive_ratio(model, X, y, n_samples=1000, bins=20):
    """
    Calculate the positive ratio of the model per bin and estimate the errors
    over these by sampling parameters from a multivariate normal distribution.

    :param model: fitted model
    :param X: [numpy.ndarray shape (n_data, n_features)] input features
    :param y: targets for fitting
    :param n_samples: number of samples to draw to calculate the error estimation
    :param bins: number of bins to distribute the model scores over

    """ 
    y_pred = model.predict(X)
    H_pred_all, bin_edges = np.histogram(y_pred, bins=bins, range=(0,1))
    H_pred_pos, _ = np.histogram(y_pred[y==1], bins=bins, range=(0,1))
    ratio = H_pred_pos.astype(float)/(H_pred_all+1e-12)
    
    p = np.random.multivariate_normal(mean=model.parameters, cov=model.cvr_mtx, size=n_samples)
    y_pred_sampled = 1./(1.+np.exp(-np.dot(p, np.concatenate((X,np.ones((X.shape[0],1))), axis=1).T)))
    H_sampled_all = np.array([np.histogram(p, bins=bins, range=(0,1))[0] for p in y_pred_sampled])
    H_sampled_pos = np.array([np.histogram(p[y==1], bins=bins, range=(0,1))[0] for p in y_pred_sampled])
    sampled_ratios = H_sampled_pos.astype(float)/(H_sampled_all+1e-12)
    err = np.sqrt(np.mean((sampled_ratios-ratio[np.newaxis,:])**2, axis=0))
    
    return ratio, err, bin_edges

def report_model_positive_ratio(model, X, y, n_samples, bins, pdf=None, pdf_name='report.pdf', figsize=(8,4)):
    """
    Create a PDF report showing the model's positive ratio verus the model score. 

    :param model: fitted model
    :param X: [numpy.ndarray shape (n_data, n_features)] input features
    :param y: targets for fitting
    :param n_samples: number of samples to draw to calculate the error estimation
    :param bins: number of bins to distribute the model scores over
    :param pdf: PDF pages object
    :param pdf_name: name of the PDF document
    :param figsize: size of the figure (tuple)
    """ 
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ratio, err, e = get_positive_ratio(model, X, y, n_samples, bins)

    ax.plot([0,1], [0,1], '-', color='black')
    ax.errorbar(x=e[:-1], y=ratio, yerr=err, fmt='o', color='orange', alpha=0.5, markersize=10, barsabove=True, capsize=10)
    ax.grid()
    ax.set_xlabel("model score")
    ax.set_ylabel("positive ratio")
    ax.set_ylim((-0.1,1.1))
    
    if pdf == None:
        pdf = PdfPages(pdf_name)
    
    pdf.savefig(fig)
    
    return pdf

def report_error_test_samples(model, X, pdf=None, pdf_name='report.pdf', figsize=(8, 4)):
    """
    Create a PDF report showing the estimated error on the provided test samples.
    These are ordered by the prediction score.

    :param model: fitted model
    :param X: [numpy.ndarray shape (n_data, n_features)] input features
    :param pdf: PDF pages object
    :param pdf_name: name of the PDF document
    :param figsize: size of the figure (tuple)
    """ 
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.linspace(0, len(X), len(X))

    y_pred = model.predict(X)
    el, eu = model.prediction_errors_from_interval(X)
    s_pred, s_el, s_eu = (np.asarray(list(t)) for t in zip(*sorted(zip(y_pred, el, eu), reverse=True)))
    ax.fill_between(x, s_pred-s_el, s_pred+s_eu, alpha=0.5, color='orange')
    ax.plot(x, s_pred, '-', color='orange')

    ax.set_ylim(0,1)
    ax.set_xlabel("Test sample (ordered by prediction score)")
    ax.set_ylabel("Prediction probability")
    ax.grid()
    
    if pdf == None:
        pdf = PdfPages(pdf_name)
    
    pdf.savefig(fig)
    
    return pdf