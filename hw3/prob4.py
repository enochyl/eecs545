from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
iris=load_iris()

# You have two features and two classifications
data_0, data_1 = iris.data[:,1:3][:50], iris.data[:,1:3][50:100]

data_t = np.r_[data_0, data_1];

# TODO: Compute the mean and covariance of each cluster, and use these to find a QDA Boundary
mean_0 = np.mean(data_0, axis=0);
mean_1 = np.mean(data_1, axis=0);
cov_0 = np.cov(data_0.T);
cov_1 = np.cov(data_1.T);
def qda(x, m, s, pi):
    x = np.matrix(x);
    m = np.matrix(m).T;
    s = np.matrix(s);
    s_inv = np.linalg.inv(s);
    return -0.5*np.log(np.linalg.det(s)) - 0.5*(x-m).T.dot(s_inv).dot(x-m) + np.log(pi);

 
# TODO: Compute the mean and covariance of the entire dataset, and use these to find a LDA Boundary
cov_t = np.cov(data_t.T);
def lda(x, m0, m1, s, pi0, pi1):
    x = np.matrix(x);
    m0 = np.matrix(m0).T;
    m0t = m0.transpose();
    m1 = np.matrix(m1).T;
    m1t = m1.transpose();
    s = np.matrix(s);
    s_inv = np.linalg.inv(s);
    return ((m1 - m0).T.dot(s_inv).dot(x) + (np.log(pi1/pi0) - 0.5*m1t.dot(s_inv).dot(m1) + 0.5*m0t.dot(s_inv).dot(m0)));

# TODO: Make two scatterplots of the data, one showing the QDA Boundary and one showing the LDA Boundary
if __name__ == "__main__":
    
    pi0 = 0.5;
    pi1 = 0.5;
    
    x = np.linspace(-1.5, 4.5, 100);
    y = np.linspace(-1, 5.5, 100);
    X, Y = np.meshgrid(x,y);
    
    qda_boundary = np.zeros((len(X), len(Y)));
    lda_boundary = np.zeros((len(X), len(Y)));
    qda_boundary0= np.zeros((len(X), 0));
    for i in range(len(X)):
        for j in range(len(Y)): 
            qda_boundary[i, j] = qda(np.c_[x[j], y[i]].T, mean_0, cov_0, pi0) - qda(np.c_[x[j], y[i]].T, mean_1, cov_1, pi1);
            lda_boundary[i, j] = lda(np.c_[x[j], y[i]].T, mean_0, mean_1, cov_t, pi0, pi1);
       
    plt.figure();
    plt.scatter(data_0[:,0], data_0[:,1]);
    plt.scatter(data_1[:,0], data_1[:,1]);
    plt.contour(X,Y, lda_boundary, 0);
    plt.title('Linear Discriminant Analysis');
    plt.xlabel('$x$');
    plt.ylabel('$y$');
    plt.grid(True);
         
    plt.figure();
    plt.scatter(data_0[:,0], data_0[:,1]);
    plt.scatter(data_1[:,0], data_1[:,1]);
    plt.contour(X,Y, qda_boundary, 0);
    plt.title('Quadratic Discriminant Analysis');
    plt.xlabel('$x$');
    plt.ylabel('$y$');
    plt.grid(True);

    
    
    