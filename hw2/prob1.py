import numpy as np
import matplotlib.pyplot as plt


# For this problem, we use data generator instead of real dataset
def data_generator(size,noise_scale=0.05):
    xs = np.random.uniform(low=0,high=3,size=size)

    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + np.sin(3*xs) + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys

# Normalize the features so that each feature has zero mean and unit standard deviation.
def featureNormalization(data):
    data_scaled = np.zeros((len(data), len(data[0])));
    for i in range(len(data[0])):
        stdev = np.std(data[:,i]);
        mean  = np.mean(data[:,i]);
        if np.abs(stdev) < 1e-8:
            data_scaled[:,i] = (data[:,i] - mean);
        else:
            data_scaled[:,i] = (data[:,i] - mean)/stdev;
    return data_scaled;

# Closed form solution
def cfs(x,y):
    x = np.c_[np.ones(len(x)), x];
    U, s, V = np.linalg.svd(x);
    S_inv = np.zeros((len(x[0]),len(x)));
    S_inv[:,:len(x[0])] = np.diag(1/s);
    x_pseudoinverse = np.dot(V.T, np.dot(S_inv, U.T));
    coef = x_pseudoinverse.dot(y);
    return coef;

# Mean-squared error
def mse(coef, soln, actual):
    N = len(soln);
    soln = np.c_[np.ones(len(soln)), soln];
    return (np.sum((np.dot(soln, coef) - actual)**2))/N;

def predicted(coef, x):
    x = np.c_[np.ones(len(x)), x];
    soln = np.dot(x, coef);
    return soln.astype(float);

# Gaussuan Kernel
def gaussianKernel(xn, x, tau):
    return np.exp((-((xn-x)**2)/(2*tau**2)));

# Locally-Weighted Linear Regression
def lwr(x, x0, y, tau):
    x  = np.c_[np.ones(len(x)), x];
    x0 = np.r_[1., x0];
    R = gaussianKernel(x, x0, tau);
    R = np.identity(len(R))*R[:,1];
    coef = np.dot(np.linalg.inv(np.dot(x.T, np.dot(R, x))), np.dot(x.T,np.dot(R,y)));
    #coef = np.dot(np.linalg.pinv(np.dot(np.sqrt(R), x)), np.dot(np.sqrt(R), y));
    return coef;

if __name__ == "__main__":
    noise_scales = [0.05,0.2]

    # for example, choose the first kind of noise scale
    noise_scale = noise_scales[0]

    # generate the data form generator given noise scale
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)
    
    # Feature normalization
    X_train_scaled = featureNormalization(X_train);
    X_test_scaled  = featureNormalization(X_test);
    
    # bandwidth parameters
    sigma_paras = [0.1,0.2,0.4,0.8,1.6]
    
    # Kernel width
    taus = [0.2, 2.0];
    
    # Linear Regression Model
    coef = cfs(X_train_scaled, y_train);
    
    lrm_error = [];
    lrm_predicted = [];
    for i in range(len(X_test)):
        lrm_predicted.append(predicted(coef,X_test_scaled[i]));
#        lrm_error.append(mse(coef, X_test_scaled[i], y_test[i]));
    
    lrm_test_error = mse(coef, X_test_scaled, y_test);
    
    plt.figure();
    plt.scatter(X_test, lrm_predicted);
    plt.grid(True);
    plt.xlabel("x");
    plt.ylabel("Predicted Labels");
    
    lwr_test_error = [];
    for tau in taus:
        lwr_error = [];
        lwr_predicted = [];
# Check error
        for i in range(len(X_test)):
            coef = lwr(X_train_scaled, X_test_scaled[i], y_train, tau);
            lwr_predicted.append(predicted(coef,X_test_scaled[i]));
            lwr_error.append(mse(coef, X_test_scaled, y_test));
        lwr_test_error.append(np.mean(lwr_error));
        
        plt.figure();
        plt.scatter(X_test, lwr_predicted);
        plt.grid(True);
        plt.xlabel("x");
        plt.ylabel("Predicted Labels");
        plt.title("$\\tau$ = %s"%tau);

#    Y_actual = X_test * 0.5 - 0.3 + np.sin(3*X_test);
#    plt.figure();
#    plt.scatter(X_test, Y_actual)
#    plt.grid(True);
#    plt.xlabel("x");
#    plt.ylabel("y");
    
    


