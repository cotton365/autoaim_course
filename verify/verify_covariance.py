"""Covariance definition / sample estimate / linear transform checks.

Run: MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_covariance.py
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-zh')
import numpy as np

def main():
    # 5-point worked example used in lesson 0001b
    P = np.array([[1.,2.],[2.,1.],[3.,4.],[4.,3.],[5.,5.]])
    mean = P.mean(0)
    X = P - mean
    N = len(P)
    S_sample = X.T @ X / (N-1)
    S_pop = X.T @ X / N
    print('points =', P.tolist())
    print('mean =', mean)
    print('deviations =', X.tolist())
    print('sum dx*dy =', float(np.sum(X[:,0]*X[:,1])))
    print('sample covariance (N-1) =', S_sample)
    print('population covariance (N) =', S_pop)
    print('corr =', S_sample[0,1]/np.sqrt(S_sample[0,0]*S_sample[1,1]))
    w, V = np.linalg.eigh(S_sample)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:,order]
    print('eigenvalues =', w, 'eigenvectors =', V.tolist())
    print('ellipse 1-sigma axis lengths =', np.sqrt(w))

    # E[XY] - E[X]E[Y] identity on the same points
    exy = np.mean(P[:,0]*P[:,1])
    print('E[XY]-E[X]E[Y] =', exy - mean[0]*mean[1])

    # Linear transform: y = A x; Cov(y) = A Cov(x) A^T
    A = np.array([[1.4, .9], [.1, 1.0]])
    rng = np.random.default_rng(20261028)
    Z = rng.normal(size=(200000, 2))
    Y = Z @ A.T
    Sy = np.cov(Y, rowvar=False, bias=True)
    target = A @ A.T
    print('\nA =', A.tolist())
    print('A A^T =', target)
    print('Monte Carlo Cov(Ax) =', Sy)
    print('max |MC - A A^T| =', float(np.max(np.abs(Sy-target))))

    # scalar version Var(aX+b)=a^2 Var(X) (sample)
    a, b = -2.5, 7.0
    x = P[:,0]
    print('\nVar(aX+b) sample =', float(np.var(a*x+b, ddof=1)))
    print('a^2 Var(X) sample =', float(a*a*np.var(x, ddof=1)))

if __name__ == '__main__':
    main()
