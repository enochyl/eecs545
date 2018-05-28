import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal


# feel free to read the two examples below, try to understand them
# in this problem, we require you to generate contour plots

# generate contour plot for function z = x^2 + 2*y^2
def plot_contour():

    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    plt.axis("square")
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)
    plt.show()


# generate heat plot (image-like) for function z = x^2 + 2*y^2
def plot_heat():
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    img = Z
    plt.imshow(img, interpolation='none', extent=[-3.0, 3.0, -3.0, 3.0],cmap="plasma")
    plt.colorbar()
    plt.axis("square")
    plt.show()


# This function receives the parameters of a multivariate Gaussian distribution
# over variables x_1, x_2 .... x_n as input and compute the marginal
def marginal_for_guassian(sigma,mu,given_indices):
    # given selected indices, compute marginal distribution for them    
    marginal_sigma = sigma[given_indices[:, None], given_indices];
    marginal_mu = mu[given_indices];
    return marginal_sigma, marginal_mu;

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
    print(mu_a)
    print(mu_b)
    
    conditional_sigma = block_aa - block_ab.dot(np.linalg.inv(block_bb)).dot(block_ba);
    conditional_mu = mu_a + block_ab.dot(np.linalg.inv(block_bb)).dot(given_values - mu_b);
    
    return conditional_sigma, conditional_mu;



if __name__ == "__main__":
    test_sigma_1 = np.array(
        [[1.0, 0.5],
         [0.5, 1.0]]
    )
    
    test_mu_1 = np.array(
        [0.0, 0.0]
    )
    
    test_sigma_2 = np.array(
        [[1.0, 0.5, 0.0, 0.0],
         [0.5, 1.0, 0.0, 1.5],
         [0.0, 0.0, 2.0, 0.0],
         [0.0, 1.5, 0.0, 4.0]]
    )
    
    test_mu_2 = np.array(
        [0.5, 0.0, -0.5, 0.0]
    )
    
    indices_1 = np.array([0])
    
    indices_2 = np.array([1,2])
    values_2 = np.array([0.1,-0.2])
    
    #plot_contour()
    #plot_heat()
    
    marginal_sigma, marginal_mu = marginal_for_guassian(test_sigma_1, test_mu_1, indices_1)
    conditional_sigma, conditional_mu = conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)
    
    # Plots
    x_marginal = np.linspace(marginal_mu - 3*np.asscalar(marginal_sigma), marginal_mu + 3*np.asscalar(marginal_sigma), 100);
    plt.figure()
    plt.plot(x_marginal,mlab.normpdf(x_marginal, marginal_mu, np.asscalar(marginal_sigma)))
    plt.xlabel('$X_1$');
    plt.ylabel('Probability distribution');
    plt.grid(True);
    
    
    x_conditional = np.linspace(conditional_mu[0] - 3*np.asscalar(conditional_sigma[0,0]), conditional_mu[0] + 3*np.asscalar(conditional_sigma[0,0]), 100);
    y_conditional = np.linspace(conditional_mu[1] - 3*np.asscalar(conditional_sigma[1,1]), conditional_mu[1] + 3*np.asscalar(conditional_sigma[1,1]), 100);
    
    X, Y = np.meshgrid(x_conditional, y_conditional);
    pos = np.empty(X.shape + (2,));
    pos[:, :, 0] = X;
    pos[:, :, 1] = Y;
    
    Z = multivariate_normal(conditional_mu, conditional_sigma);
    plt.figure()
    plt.contourf(X, Y, Z.pdf(pos), 50);
    plt.xlabel('$X_1$');
    plt.ylabel('$X_4$');
    plt.colorbar();
    




















