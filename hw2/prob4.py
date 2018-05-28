import numpy as np
import matplotlib.pyplot as plt


def computeKernel(x, xprime, sigma):
    return np.exp(-((x-xprime)**2)/2/sigma**2);

def jointDistribution(x, sigma):
    X, Y = np.meshgrid(x, x);
    cov = computeKernel(X,Y,sigma);
    mu = np.zeros(len(x));
    return mu, cov

def conditional_for_gaussian(sigma,mu,given_indices,given_values):
    # given some indices that have fixed value, compute the conditional distribution
    # for rest indices
    
    complement_indices = np.asarray(range(len(sigma)));
    complement_indices = np.delete(complement_indices, given_indices);
    
    block_aa = sigma[complement_indices[:, None], complement_indices];
    block_bb = sigma[     given_indices[:, None],      given_indices];
    block_ba = sigma[     given_indices[:, None], complement_indices];
    block_ab = sigma[complement_indices[:, None],      given_indices];
    
    mu_a = mu[complement_indices];
    mu_b = mu[given_indices];
    
    conditional_sigma = block_aa - block_ab.dot(np.linalg.inv(block_bb)).dot(block_ba);
    conditional_mu = mu_a + block_ab.dot(np.linalg.inv(block_bb)).dot(given_values - mu_b);
    
    return conditional_mu, conditional_sigma;


if __name__ == "__main__":
    x = np.linspace(-5,5,100);
    sigmas = [0.3, 0.5, 1.0];
    
    for sigma in sigmas:
        mu, cov = jointDistribution(x, sigma);
        sample = np.random.multivariate_normal(mu, cov, 5);
        plt.figure();
        for i in range(len(sample)):
            plt.plot(x, sample[i]);
            plt.grid(True);
            plt.xlabel('$x$');
            plt.ylabel('Sample Distribution');
            plt.title("$\sigma$ = %s"%sigma);
        
    D = np.asarray([[-1.3, 2], [2.4, 5.2], [-2.5, -1.5], [-3.3, -0.8], [0.3, 0.3]]);
    
    xc = np.append(x, D[:,0]);
    
    for sigma in sigmas:
        mu, cov = jointDistribution(xc, sigma);
        mu_c, cov_c = conditional_for_gaussian(cov,mu,np.asarray([100, 101, 102, 103, 104]),D[:,1]);
        sample = np.random.multivariate_normal(mu_c, cov_c, 5);
        plt.figure();
        for i in range(len(sample)):
            plt.plot(x, sample[i], linewidth=0.8);
            plt.plot(x, mu_c, color='black');
            plt.grid(True);
            plt.xlabel('$x$');
            plt.ylabel('Sample Distribution');
            plt.title("$\sigma$ = %s"%sigma);
        plt.scatter(D[:,0], D[:,1], color='black');
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    