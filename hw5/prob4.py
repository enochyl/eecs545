import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
np.random.seed(1234)

tol = 1e-5;

def random_posdef(n):
  A = np.random.rand(n, n)
  return np.dot(A, A.transpose())

def plotMixture(x, y, data, means, covs, K):
    plt.figure();
    plt.scatter(data[:,0], data[:,1]);
    plt.grid('on');
    X,Y = np.meshgrid(x,y);
    for k in range(K):
        mixture = np.zeros((len(X), len(Y)));
        for i in range(len(X)):
            for j in range(len(Y)):
                mixture[i,j] = multivariate_normal.pdf(np.c_[x[j],y[i]], means[k], covs[k]);
        plt.contour(X, Y, mixture, 5);
        plt.title('K = %i' %K);
    plt.show();


def gaussianMixture(x, y, K, data, iterations, mode):
    # Parameter initialization ###
    pi = [1.0/K for i in range(K)]
    means = [[0,0] for i in range(K)]
    covs = [random_posdef(2) for i in range(K)]
    N = len(data);
    ##############################
    for i in range(iterations):

        # E Step
        gamma = np.zeros((N,K));
        for k in range(K):
            for n in range(N):
                gamma[n,k] = pi[k]*multivariate_normal.pdf(data[n], mean = means[k], cov = covs[k]);
        gamma = (gamma.T / gamma.sum(axis=1)).T;

        # M Step
        Nk = np.sum(gamma, axis=0);
        means_new = (np.dot(gamma.T, data).T/Nk).T;
        covs_new = [];
        for k in range(K):
            temp = 0;
            for n in range(N):
                temp += gamma[n,k]*np.matmul(np.asmatrix(data[n]-means_new[k]).T, np.asmatrix(data[n]-means_new[k]));
            covs_new.append(temp/Nk[k]);
        pi_new = Nk/N;

        if mode == 1 and (i == 0 or i == 4 or i == 9 or i == 19 or i == 49):
            plotMixture(x, y, data, means, covs, K);

        # Update
        means = means_new;
        covs = covs_new;
        pi = pi_new;
    print('pi values:')
    print(pi);
    print('mean values:');
    print(means);
    print('covariance values:');
    print(covs);

    return means, covs;


if __name__ == "__main__":

    data = np.load('gmm_data.npy');
    Ks = [2, 3, 5, 10];
    x = np.linspace(-2, 7, 100);
    y = np.linspace(-1, 9, 100);
    for K in Ks:
        [means, covs] = gaussianMixture(x, y, K, data, 50, 0);
        plotMixture(x, y, data, means, covs, K);

    gaussianMixture(x, y, 3, data, 50, 1);




# [ 0.29085371  0.41871864  0.29042766]
#
# [[ 1.05523113  2.85999485]
#  [ 2.96030513  5.00699832]
#  [ 4.05189929  2.48155162]]
#
# [matrix([[ 0.62283072,  0.36501207],
#         [ 0.36501207,  0.53719646]]), matrix([[ 1.01032542, -0.01279441],
#         [-0.01279441,  1.05066972]]), matrix([[ 0.21541969, -0.36190934],
#         [-0.36190934,  1.12336943]])]
#
#
#
