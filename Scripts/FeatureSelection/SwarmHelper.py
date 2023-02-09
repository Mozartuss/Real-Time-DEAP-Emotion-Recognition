import math

import numpy as np
from numpy.random import rand

import cpuinfo

info = cpuinfo.get_cpu_info()
vendor = info['vendor_id_raw']
if vendor == "GenuineIntel":
    from sklearnex import patch_sklearn
    patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier


def error_rate(x, opts):
    # parameters
    k = opts['k']
    fold = opts['fold']
    xt = fold['xt']
    yt = fold['yt']
    xv = fold['xv']
    yv = fold['yv']

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain = xt[:, x == 1]
    ytrain = yt.reshape(num_train)  # Solve bug
    xvalid = xv[:, x == 1]
    yvalid = yv.reshape(num_valid)  # Solve bug
    # Training
    # mdl = SVC(max_iter=500, gamma='auto', random_state=10, kernel="rbf")
    mdl = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred = mdl.predict(xvalid)
    acc = np.sum(yvalid == ypred) / num_valid
    error = 1 - acc

    return error



# Error rate & Feature size
def Fun(x, opts):
    # Parameters
    alpha = 0.99
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1
    else:
        # Get error rate
        error = error_rate(x, opts)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X


def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]

    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()

    return V, Vmax, Vmin


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def transfer_function(x):
    Xs = abs(np.tanh(x))

    return Xs


# Levy Flight
def levy_distribution(beta, dim):
    # Sigma
    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)
    # Parameter u & v
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    # Step
    step = u / abs(v) ** (1 / beta)
    LF = 0.01 * step

    return LF


# --- Opposition based learning (7)
def opposition_based_learning(X, lb, ub, N, dim):
    Xo = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            Xo[i, d] = lb[0, d] + ub[0, d] - X[i, d]

    return Xo


def cs_optimizer(xtrain, opts, resulting_channels, pbar):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    Pa = 0.25  # discovery rate
    alpha = 1  # constant
    beta = 1.5  # levy component

    N = opts['N']
    max_iter = opts['T']
    if 'Pa' in opts:
        Pa = opts['Pa']
    if 'alpha' in opts:
        alpha = opts['alpha']
    if 'beta' in opts:
        beta = opts['beta']

        # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')

    for i in range(N):
        fit[i, 0] = Fun(Xbin[i, :], opts)
        if fit[i, 0] < fitG:
            Xgb[0, :] = X[i, :]
            fitG = fit[i, 0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = fitG.copy()
    t += 1

    while t < max_iter:
        Xnew = np.zeros([N, dim], dtype='float')

        # {1} Random walk/Levy flight phase
        for i in range(N):
            # Levy distribution
            L = levy_distribution(beta, dim)
            for d in range(dim):
                # Levy flight (1)
                Xnew[i, d] = X[i, d] + alpha * L[d] * (X[i, d] - Xgb[0, d])
                # Boundary
                Xnew[i, d] = boundary(Xnew[i, d], lb[0, d], ub[0, d])

        # Binary conversion
        Xbin = binary_conversion(Xnew, thres, N, dim)

        # Greedy selection
        for i in range(N):
            Fnew = Fun(Xbin[i, :], opts)
            if Fnew <= fit[i, 0]:
                X[i, :] = Xnew[i, :]
                fit[i, 0] = Fnew

            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0]

        # {2} Discovery and abandon worse nests phase
        J = np.random.permutation(N)
        K = np.random.permutation(N)
        Xj = np.zeros([N, dim], dtype='float')
        Xk = np.zeros([N, dim], dtype='float')
        for i in range(N):
            Xj[i, :] = X[J[i], :]
            Xk[i, :] = X[K[i], :]

        Xnew = np.zeros([N, dim], dtype='float')

        for i in range(N):
            Xnew[i, :] = X[i, :]
            r = rand()
            for d in range(dim):
                # A fraction of worse nest is discovered with a probability
                if rand() < Pa:
                    Xnew[i, d] = X[i, d] + r * (Xj[i, d] - Xk[i, d])

                # Boundary
                Xnew[i, d] = boundary(Xnew[i, d], lb[0, d], ub[0, d])

        # Binary conversion
        Xbin = binary_conversion(Xnew, thres, N, dim)

        # Greedy selection
        for i in range(N):
            Fnew = Fun(Xbin[i, :], opts)
            if Fnew <= fit[i, 0]:
                X[i, :] = Xnew[i, :]
                fit[i, 0] = Fnew

            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0]

        # Store result
        curve[0, t] = fitG.copy()
        t += 1
        pbar.update(1)

        # Best feature subset
    return Xgb[0].argsort()[-resulting_channels:][::-1]


def bat_optimizer(xtrain, opts, resulting_channels, pbar):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    fmax = 2  # maximum frequency
    fmin = 0  # minimum frequency
    alpha = 0.9  # constant
    gamma = 0.9  # constant
    A_max = 2  # maximum loudness
    r0_max = 1  # maximum pulse rate

    N = opts['N']
    max_iter = opts['T']
    if 'fmax' in opts:
        fmax = opts['fmax']
    if 'fmin' in opts:
        fmin = opts['fmin']
    if 'alpha' in opts:
        alpha = opts['alpha']
    if 'gamma' in opts:
        gamma = opts['gamma']
    if 'A' in opts:
        A_max = opts['A']
    if 'r' in opts:
        r0_max = opts['r']
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position & velocity
    X = init_position(lb, ub, N, dim)
    V = np.zeros([N, dim], dtype='float')

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')

    for i in range(N):
        fit[i, 0] = Fun(Xbin[i, :], opts)
        if fit[i, 0] < fitG:
            Xgb[0, :] = X[i, :]
            fitG = fit[i, 0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = fitG.copy()
    t += 1

    # Initial loudness [1 ~ 2] & pulse rate [0 ~ 1]
    A = np.random.uniform(1, A_max, N)
    r0 = np.random.uniform(0, r0_max, N)
    r = r0.copy()

    while t < max_iter:
        Xnew = np.zeros([N, dim], dtype='float')

        for i in range(N):
            # beta [0 ~1]
            beta = rand()
            # frequency (2)
            freq = fmin + (fmax - fmin) * beta
            for d in range(dim):
                # Velocity update (3)
                V[i, d] = V[i, d] + (X[i, d] - Xgb[0, d]) * freq
                # Position update (4)
                Xnew[i, d] = X[i, d] + V[i, d]
                # Boundary
                Xnew[i, d] = boundary(Xnew[i, d], lb[0, d], ub[0, d])

            # Generate local solution around best solution
            if rand() > r[i]:
                for d in range(dim):
                    # Epsilon in [-1,1]
                    eps = -1 + 2 * rand()
                    # Random walk (5)
                    Xnew[i, d] = Xgb[0, d] + eps * np.mean(A)
                    # Boundary
                    Xnew[i, d] = boundary(Xnew[i, d], lb[0, d], ub[0, d])

        # Binary conversion
        Xbin = binary_conversion(Xnew, thres, N, dim)

        # Greedy selection
        for i in range(N):
            Fnew = Fun(Xbin[i, :], opts)
            if (rand() < A[i]) and (Fnew <= fit[i, 0]):
                X[i, :] = Xnew[i, :]
                fit[i, 0] = Fnew
                # Loudness update (6)
                A[i] = alpha * A[i]
                # Pulse rate update (6)
                r[i] = r0[i] * (1 - np.exp(-gamma * t))

            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0]

        # Store result
        curve[0, t] = fitG.copy()
        t += 1
        pbar.update(1)

        # Best feature subset
    return Xgb[0].argsort()[-resulting_channels:][::-1]


def gwo_optimizer(xtrain, opts, resulting_channels, pbar):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5

    N = opts['N']
    max_iter = opts['T']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')

    for i in range(N):
        fit[i, 0] = Fun(Xbin[i, :], opts)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]

        if Fbeta > fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]

        if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    t += 1

    while t < max_iter:
        # Coefficient decreases linearly from 2 to 0
        a = 2 - t * (2 / max_iter)

        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                # Parameter A (3.3)
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6)
                X1 = Xalpha[0, d] - A1 * Dalpha
                X2 = Xbeta[0, d] - A2 * Dbeta
                X3 = Xdelta[0, d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i, d] = (X1 + X2 + X3) / 3
                # Boundary
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(Xbin[i, :], opts)
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]

            if Fbeta > fit[i, 0] > Falpha:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]

            if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]

        curve[0, t] = Falpha.copy()
        t += 1
        pbar.update(1)

    # Best feature subset
    return Xalpha[0].argsort()[-resulting_channels:][::-1]


def tmgwo_optimizer(xtrain, opts, resulting_channels, pbar):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    Mp = 0.5  # mutation probability

    N = opts['N']
    max_iter = opts['T']
    if 'Mp' in opts:
        Mp = opts['Mp']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # --- Binary conversion
    X = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='int')
    Xbeta = np.zeros([1, dim], dtype='int')
    Xdelta = np.zeros([1, dim], dtype='int')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')

    for i in range(N):
        fit[i, 0] = Fun(X[i, :], opts)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]
        if Fbeta > fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]
        if Fdelta > fit[i, 0] > Falpha and fit[i, 0] > Fbeta:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    t += 1

    while t < max_iter:
        # Coefficient decreases linearly from 2 to 0 (3.5)
        a = 2 - t * (2 / max_iter)

        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.7 - 3.9)
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                # Parameter A (3.3)
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.7 -3.9)
                X1 = Xalpha[0, d] - A1 * Dalpha
                X2 = Xbeta[0, d] - A2 * Dbeta
                X3 = Xdelta[0, d] - A3 * Ddelta
                # Update wolf (3.6)
                Xn = (X1 + X2 + X3) / 3
                # --- transfer function
                Xs = transfer_function(Xn)
                # --- update position (4.3.2)
                if rand() < Xs:
                    X[i, d] = 0
                else:
                    X[i, d] = 1

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(X[i, :], opts)
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]
            if Fbeta > fit[i, 0] > Falpha:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]
            if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]

        curve[0, t] = Falpha.copy()
        t += 1

        # --- two phase mutation: first phase
        # find index of 1
        idx = np.where(Xalpha == 1)
        idx1 = idx[1]
        Xmut1 = np.zeros([1, dim], dtype='int')
        Xmut1[0, :] = Xalpha[0, :]
        for d in range(len(idx1)):
            r = rand()
            if isinstance(r, tuple):
                r = float('.'.join(str(elem) for elem in r))
            if r < Mp:
                Xmut1[0, idx1[d]] = 0
                Fnew1 = Fun(Xmut1[0, :], opts)
                if Fnew1 < Falpha:
                    Falpha = Fnew1
                    Xalpha[0, :] = Xmut1[0, :]

        # --- two phase mutation: second phase
        # find index of 0
        idx = np.where(Xalpha == 0)
        idx0 = idx[1]
        Xmut2 = np.zeros([1, dim], dtype='int')
        Xmut2[0, :] = Xalpha[0, :]
        for d in range(len(idx0)):
            r = rand()
            if isinstance(r, tuple):
                r = float('.'.join(str(elem) for elem in r))
            if r < Mp:
                Xmut2[0, idx0[d]] = 1
                Fnew2 = Fun(Xmut2[0, :], opts)
                if Fnew2 < Falpha:
                    Falpha = Fnew2
                    Xalpha[0, :] = Xmut2[0, :]
        pbar.update(1)

    return Xalpha[0].argsort()[-resulting_channels:][::-1]


def issa_optimizer(xtrain, opts, resulting_channels, pbar):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    max_local_iter = 10  # maximum iteration for local search

    N = opts['N']
    max_iter = opts['T']
    if 'maxLt' in opts:
        max_local_iter = opts['maxLt']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xf = np.zeros([1, dim], dtype='float')
    fitF = float('inf')

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness
    for i in range(N):
        fit[i, 0] = Fun(Xbin[i, :], opts)
        if fit[i, 0] < fitF:
            Xf[0, :] = X[i, :]
            fitF = fit[i, 0]

    # --- Opposition based learning
    Xo = opposition_based_learning(X, lb, ub, N, dim)
    # --- Binary conversion
    Xobin = binary_conversion(Xo, thres, N, dim)

    # --- Fitness
    fitO = np.zeros([N, 1], dtype='float')
    for i in range(N):
        fitO[i, 0] = Fun(Xobin[i, :], opts)
        if fitO[i, 0] < fitF:
            Xf[0, :] = Xo[i, :]
            fitF = fitO[i, 0]

    # --- Merge opposite & current population, and select best N
    XX = np.concatenate((X, Xo), axis=0)
    FF = np.concatenate((fit, fitO), axis=0)
    # --- Sort in ascending order
    ind = np.argsort(FF, axis=0)
    for i in range(N):
        X[i, :] = XX[ind[i, 0], :]
        fit[i, 0] = FF[ind[i, 0]]

    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    # Store result
    curve[0, t] = fitF.copy()
    t += 1

    while t < max_iter:
        # Compute coefficient, c1 (2)
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)

        for i in range(N):
            # First leader update
            if i == 0:
                for d in range(dim):
                    # Coefficient c2 & c3 [0 ~ 1]
                    c2 = rand()
                    c3 = rand()
                    # Leader update (1)
                    if c3 >= 0.5:
                        X[i, d] = Xf[0, d] + c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])
                    else:
                        X[i, d] = Xf[0, d] - c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])

                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    # Salp update
            elif i >= 1:
                for d in range(dim):
                    # Salp update by following front salp (3)
                    X[i, d] = (X[i, d] + X[i - 1, d]) / 2
                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(Xbin[i, :], opts)
            if fit[i, 0] < fitF:
                Xf[0, :] = X[i, :]
                fitF = fit[i, 0]

        # --- Local search algorithm
        Lt = 0
        temp = np.zeros([1, dim], dtype='float')
        temp[0, :] = Xf[0, :]

        while Lt < max_local_iter:
            # --- Random three features
            RD = np.random.permutation(dim)
            for d in range(3):
                index = RD[d]
                # --- Flip the selected three features
                if temp[0, index] > thres:
                    temp[0, index] = temp[0, index] - thres
                else:
                    temp[0, index] = temp[0, index] + thres

            # --- Binary conversion
            temp_bin = binary_conversion(temp, thres, 1, dim)

            # --- Fitness
            Fnew = Fun(temp_bin[0, :], opts)
            if Fnew < fitF:
                fitF = Fnew
                Xf[0, :] = temp[0, :]

            Lt += 1

        # Store result
        curve[0, t] = fitF.copy()
        t += 1
        pbar.update(1)

    return Xf[0].argsort()[-resulting_channels:][::-1]


def ssa_optimizer(xtrain, opts, resulting_channels, pbar):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5

    N = opts['N']
    max_iter = opts['T']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xf = np.zeros([1, dim], dtype='float')
    fitF = float('inf')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:

        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(Xbin[i, :], opts)
            if fit[i, 0] < fitF:
                Xf[0, :] = X[i, :]
                fitF = fit[i, 0]

        # Store result
        curve[0, t] = fitF.copy()
        t += 1

        # Compute coefficient, c1 (3.2)
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)

        for i in range(N):
            # First leader update
            if i == 0:
                for d in range(dim):
                    # Coefficient c2 & c3 [0 ~ 1]
                    c2 = rand()
                    c3 = rand()
                    # Leader update (3.1)
                    if c3 >= 0.5:
                        X[i, d] = Xf[0, d] + c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])
                    else:
                        X[i, d] = Xf[0, d] - c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])

                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    # Salp update
            elif i >= 1:
                for d in range(dim):
                    # Salp update by following front salp (3.4)
                    X[i, d] = (X[i, d] + X[i - 1, d]) / 2
                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
        pbar.update(1)

    return Xf[0].argsort()[-resulting_channels:][::-1]


def pso_optimizer(xtrain, opts, resulting_channels, pbar):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9  # inertia weight
    c1 = 2  # acceleration factor
    c2 = 2  # acceleration factor

    N = opts['N']
    max_iter = opts['T']
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']

        # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position & velocity
    X = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(Xbin[i, :], opts)
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]

        # Store result
        curve[0, t] = fitG.copy()
        t += 1

        for i in range(N):
            for d in range(dim):
                # Update velocity
                r1 = rand()
                r2 = rand()
                V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xgb[0, d] - X[i, d])
                # Boundary
                V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                # Update position
                X[i, d] = X[i, d] + V[i, d]
                # Boundary
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
        pbar.update(1)

    # Best feature subset
    return Xgb[0].argsort()[-resulting_channels:][::-1]
