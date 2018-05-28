import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

# Defining new polynomial features
def basis(X_train, order):
    if order == 0:
        return np.ones((len(X_train), 1));
    newFeatureMatrix = np.zeros((len(X_train), order*(len(X_train[0]))));
    k = -1;
    for i in range(len(X_train[0])):
        for j in range(order):
            k += 1;
            newFeatureMatrix[:,k] = (X_train[:,i])**(j+1);
    return newFeatureMatrix;
    
# Closed form solution
def cfs(X_train, y_train):
    attri = np.c_[np.ones(len(X_train)), X_train];
    U, s, V = np.linalg.svd(attri);
    S_inv = np.zeros((len(attri[0]),len(attri)));
    S_inv[:,:len(attri[0])] = np.diag(1/s);
    x_pseudoinverse = np.dot(V.T, np.dot(S_inv, U.T));
    coef = x_pseudoinverse.dot(y_train);
    return coef;

# Root mean-squared error
def rmse(coef, soln, actual):
    featureVector = np.c_[np.ones(len(soln)), soln];
    return np.sqrt(np.mean((np.dot(featureVector, coef) - actual)**2));

# Main function
if __name__ == "__main__":
    partitions = [0.2, 0.4, 0.6, 0.8, 1.0];
    orderVector = [0, 1, 2, 3, 4];
    error_train, error_test = [], [];
    error_train_partitioned, error_test_partitioned = [], [];
    for order in orderVector:
        # Generate new features based on order of polynomial
        newFeatureMatrix_train = basis(X_train, order);
        newFeatureMatrix_test  = basis( X_test, order);
        # Compute the weight vector
        coef = cfs(newFeatureMatrix_train, y_train).T;
        # Compute the root mean-squared error of test and training set
        error_train.append(rmse(coef, newFeatureMatrix_train, y_train));
        error_test.append(rmse(coef, newFeatureMatrix_test, y_test));
    
    for partition in partitions:
        Nsplit_partition = round(len(X_train)*partition);
        if partition != 1:
            X_train_partitioned, y_train_partitioned = X_train[0:Nsplit_partition], y_train[0:Nsplit_partition];
        else:
            X_train_partitioned, y_train_partitioned = X_train, y_train;
        newFeatureMatrix_train = basis(X_train_partitioned, 1);
        newFeatureMatrix_test  = basis(             X_test, 1);
        coef = cfs(newFeatureMatrix_train, y_train_partitioned);
        error_train_partitioned.append(rmse(coef, newFeatureMatrix_train, y_train_partitioned));
        error_test_partitioned.append(rmse(coef, newFeatureMatrix_test, y_test));
        
    plt.figure();
    plt.plot(orderVector, error_train, label="Training Error");
    plt.plot(orderVector,  error_test, label="Test Error");
    plt.grid(True);
    plt.xlabel("Order");
    plt.ylabel("Error");
    plt.legend();
    
    plt.figure();
    plt.plot(partitions, error_train_partitioned, label="Training Error");
    plt.plot(partitions,  error_test_partitioned, label="Test Error");
    plt.grid(True)
    plt.xlabel("Percentage partition");
    plt.ylabel("Error");
    plt.legend();
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    