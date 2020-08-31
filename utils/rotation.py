
import numpy as np
import scipy as sp


def structured_varimax(U, n_timeseries, window, gamma=1, tol=1e-8, max_iter=5000):
    # Borrowed from https://github.com/kieferk/pymssa
    # See:
    # http://200.145.112.249/webcast/files/SeminarMAR2017-ICTP-SAIFR.pdf

    # get the shape of the singular vectors
    p, k = U.shape

    # initialize the varimax rotation to identity matrix
    T = np.eye(k)

    # initialize singular value sum tracker
    d = 0

    # rename to match variable names in code referenced in paper above
    # for clarity
    D = n_timeseries
    M = window

    # initialize matrices
    vec_i = np.ones(M).reshape((1, M))
    I_d = np.eye(D)

    # kronecker product
    I_d_md = np.kron(I_d, vec_i)

    M = I_d - (gamma / D) * np.ones((D, D))
    IMI = np.dot(I_d_md.T, np.dot(M, I_d_md))

    d_old = 0
    iteration = 0

    while (d_old == 0) or ((d / d_old) > (1 + tol)):

        d_old = d
        iteration = iteration + 1

        B = np.dot(U, T)
        G = np.dot(U.T, (B * np.dot(IMI, B ** 2)))

        u, s, vh = np.linalg.svd(G)

        T = np.dot(u, vh)
        d = np.sum(s)

        if iteration >= max_iter:
            break

    return T


def varimax(loadings, normalize=True, max_iter=1000, tol=1e-5):
    # Borrowed from:
    # https://github.com/EducationalTestingService/factor_analyzer/blob/master/factor_analyzer
    """
    Perform varimax (orthogonal) rotation, with optional
    Kaiser normalization.
    Parameters
    ----------
    loadings : array-like
        The loading matrix
    Returns
    -------
    loadings : numpy array, shape (n_features, n_factors)
        The loadings matrix
    rotation_mtx : numpy array, shape (n_factors, n_factors)
        The rotation matrix
    """
    X = loadings.copy()
    n_rows, n_cols = X.shape
    if n_cols < 2:
        return X

    # normalize the loadings matrix
    # using sqrt of the sum of squares (Kaiser)
    if normalize:
        normalized_mtx = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 1, X.copy())
        X = (X.T / normalized_mtx).T

    # initialize the rotation matrix
    # to N x N identity matrix
    rotation_mtx = np.eye(n_cols)

    d = 0
    for _ in range(max_iter):

        old_d = d
        # take inner product of loading matrix
        # and rotation matrix
        basis = np.dot(X, rotation_mtx)

        # transform data for singular value decomposition
        transformed = np.dot(X.T, basis**3 - (1.0 / n_rows) *
                             np.dot(basis, np.diag(np.diag(np.dot(basis.T, basis)))))

        # perform SVD on
        # the transformed matrix
        U, S, V = np.linalg.svd(transformed)

        # take inner product of U and V, and sum of S
        rotation_mtx = np.dot(U, V)
        d = np.sum(S)

        # check convergence
        if old_d != 0 and d / old_d < 1 + tol:
            break

    # take inner product of loading matrix
    # and rotation matrix
    X = np.dot(X, rotation_mtx)

    # de-normalize the data
    if normalize:
        X = X.T * normalized_mtx
    else:
        X = X.T

    # convert loadings matrix to data frame
    loadings = X.T.copy()
    return loadings, rotation_mtx
