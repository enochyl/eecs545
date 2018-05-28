import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# we defined a class for sequential bayesian learner
class bayesian_linear_regression(object):

    # initialized with covariance matrix(sigma), mean vector(mu) and prior(beta)
    def __init__(self,sigma,mu,beta):
        self.prior = multivariate_normal(mean = mu, cov = sigma);
        self.sigma = sigma
        self.mu = mu
        self.beta = beta
        
        self.sigma_N = self.sigma;
        self.mu_N = self.mu;
        self.posterior = self.prior;

    # you need to implement the update function
    # when received additional design matrix phi and continuous label t
    def update(self,phi,t):
                
        self.sigma_N = np.linalg.inv(np.linalg.inv(self.sigma) + self.beta*phi.T.dot(phi));
        self.mu_N = self.sigma_N.dot((self.beta*phi.T.dot(t) + np.linalg.inv(self.sigma).dot(self.mu)));
        self.posterior = multivariate_normal(mean=self.mu_N, cov=self.sigma_N);
        
        self.sigma = self.sigma_N;
        self.mu = self.mu_N;
        
        pass

    def contourPlot(self, x, y, weight = []):
        print(self.sigma)
        print(self.mu)
        
        pos = np.empty(x.shape + (2,));
        pos[:, :, 0] = x;
        pos[:, :, 1] = y;
        
        plt.figure()
        plt.contourf(x, y, self.posterior.pdf(pos), 20);
        plt.xlabel('$w_0$');
        plt.ylabel('$w_1$');
        plt.colorbar();
        
        if weight:
            plt.scatter(weight[0], weight[1], marker = '+', c = 'black', s = 60);

def data_generator(size,scale):
    x = np.random.uniform(low=-3, high=3, size=size)
    rand = np.random.normal(0, scale=scale, size=size)
    y = 0.5 * x - 0.3 + rand
    phi = np.array([[x[i], 1] for i in range(x.shape[0])])
    t = y
    return phi, t

if __name__ == "__main__":
    # initialization
    alpha = 2;
    sigma_0 = np.diag(1.0/alpha*np.ones([2]))
    mu_0 = np.zeros([2])
    beta = 1.0
    x, y = np.mgrid[-1:1:.01, -1:1:.01];
    blr_learner = bayesian_linear_regression(sigma_0, mu_0, beta=beta)
    blr_learner.contourPlot(x, y, weight = [0.5, -0.3]);
    num_episodes = 20
    for epi in range(num_episodes):
        phi, t = data_generator(1,1.0/beta);
        blr_learner.update(phi,t);
        if epi == 0 or epi == 4 or epi == 19:
            blr_learner.contourPlot(x, y, weight = [0.5, -0.3]);
        


