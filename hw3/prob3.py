import numpy as np
import matplotlib.pyplot as plt

np.random.seed(17)
# data and target are given below 
# data is a numpy array consisting of 100 2-dimensional points
# target is a numpy array consisting of 100 values of 1 or -1
data = np.ones((100, 2))
data[:,0] = np.random.uniform(-1.5, 1.5, 100)
data[:,1] = np.random.uniform(-2, 2, 100)
z = data[:,0] ** 2 + ( data[:,1] - (data[:,0] ** 2) ** 0.333 ) ** 2  
target = np.asarray( z > 1.5, dtype = int) * 2 - 1

def gaussianKernel(x, xprime, sigma):
        return np.exp(-(np.linalg.norm(x-xprime)**2)/(2*(sigma**2)));

def kernelizedPerceptron(train, y, n_epoch, sigma):
    a = np.zeros((len(train)));
    k = np.zeros((len(train), len(train)));
    for i in range(len(train)):
        for j in range(len(train)):
            k[i,j] = gaussianKernel(train[i,:], train[j,:], sigma);

    for epoch in range(n_epoch):
        for j in range(len(train)):
            if np.sign(np.sum(k[:,j] * a * y)) != y[j]:
                a[j] += 1.0;      
    return a, k;


if __name__ == "__main__":
    
    sigmas = [0.1, 1.0];
    for sigma in sigmas:
        a, k = kernelizedPerceptron(data, target, 1000, sigma);
        xaxis = np.linspace(-1.5, 1.5, 100);
        yaxis = np.linspace(-2.0, 2.0, 100);
        X, Y = np.meshgrid(xaxis,yaxis);
        
        boundary = np.zeros((len(X),len(Y)));
                
        for i in range(len(X)):
            for j in range(len(Y)):
                kn = np.exp(-(np.linalg.norm(data-np.c_[X[i,j],Y[i,j]], axis = 1)**2)/(2*(sigma**2)));
                boundary[i,j] = (np.sum(kn * a * target)) ;
        
        plt.figure();
        plt.scatter(data[:,0], data[:,1], c = target);
        plt.contour(X,Y,boundary, 0);
        plt.xlabel('$x_0$');
        plt.ylabel('$x_1$');
        plt.title("$\sigma$ = %s"%sigma);
        plt.grid(True);
    
    
    
    
