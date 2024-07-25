import numpy as np
import scipy as sp

def damping(t, x, N=None, threshold=0.1, rho=100.0):
    """
    Computes minimum damping via the matrix pencil method.

    Implementation is from Jacobson et al 2019.

    Parameters
    ----------
    t : (k,) ndarray
        time
    x : (k,) ndarray
        response
    N : int
        downsample via cubic spline interpolation, optional
    threshold : float in (0.0, 1.0)
        fraction of largest singular value below which model order is cut off
    rho : float
        aggregation parameter for KS function to approximate minimum damping
    
    Returns
    -------
    alpha_ks : float
        minimum damping (maximum growth) of signal
    alphas : (M,) ndarray
        list of damping values
    omegas : (M,) ndarray
        list of frequencies
    """

    # 1. Downsample via interpolation
    # N = max(N, int(len(t)/10))
    if N is None:
        N = len(t)
    cs = sp.interpolate.CubicSpline(t, x)
    ts = np.linspace(t[0], t[-1], N)
    Z = cs(ts)
    dt = ts[1] - ts[0]

    # 2. Set pencil parameter
    L = int(N/2 - 1)

    # 3. Fill Hankel matrix Y with samples Z
    Y = np.zeros((N-L, L+1))
    for i in np.arange(0, N-L):
        for j in np.arange(0, L+1):
            Y[i,j] = Z[i+j]
    
    # 4. Singular value decomposition
    U, S, Vt = np.linalg.svd(Y)

    # 5. Choose model order M
    smax = np.max(S)
    Snew = []
    for scurr in S:
        if scurr < threshold * smax:
            break
        Snew.append(scurr)
    M = len(Snew)
    # print(f"M = {M:d}")

    # 6. Compute A matrix
    V = Vt.T
    Vhat = V[:,:M]
    Vhat1 = Vhat[:L, :]
    Vhat2 = Vhat[1:L+1, :]
    Vhat1t_pinv = np.linalg.pinv(Vhat1.T)
    Vhat2t = Vhat2.T
    A = Vhat1t_pinv @ Vhat2t

    # 7. Eigendecomposition
    lambdas = np.linalg.eigvals(A)
    lambdahat = lambdas[:M]

    # 8. Compute damping and frequency
    sks = np.log(lambdahat)
    alphas = np.real(sks)/dt
    omegas = np.imag(sks)/dt
    # alpha_ks = (1/rho) * np.log(np.sum(np.exp(rho*alphas)))
    alpha_ks = np.max(alphas) + (1/rho) * np.log(np.sum(np.exp(rho*(alphas-np.max(alphas)))))

    # 9. Compute amplitude
    # Z2 = np.zeros((L, L), dtype=complex)
    # for i in np.arange(0,L):  # L=len(lambdas)
    #     Z2[:,i] = np.power(lambdas[i], np.arange(0,L), dtype=complex).T
    # h = np.linalg.lstsq(Z2, Z[:L], rcond=None)
    # h = h[0]

    # return alpha_ks, omegas, alphas, np.imag(h[:M])
    return float(alpha_ks), alphas, omegas
