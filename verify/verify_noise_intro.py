"""Numeric check for the navigation example in lesson 0003."""
import math
from scipy.stats import chi2

def main():
    x_prior, P_prior = 100.0, 25.0
    z, R = 108.0, 4.0
    S = P_prior + R
    K = P_prior / S
    x_post = x_prior + K * (z - x_prior)
    P_post = (1 - K) * P_prior
    NIS = (z - x_prior) ** 2 / S
    print(f'S={S:.6f}')
    print(f'K={K:.6f}')
    print(f'x+={x_post:.6f}')
    print(f'P+={P_post:.6f}')
    print(f'sigma+={math.sqrt(P_post):.6f}')
    print(f'NIS={NIS:.6f}')
    print(f'chi2_1_95={chi2.ppf(0.95, 1):.6f}')
    assert abs(K - 25/29) < 1e-12
    assert abs(x_post - 106.89655172413794) < 1e-9
    assert abs(P_post - 100/29) < 1e-12
    assert abs(NIS - 64/29) < 1e-12
    print('ALL CHECKS PASSED')

if __name__ == '__main__':
    main()
