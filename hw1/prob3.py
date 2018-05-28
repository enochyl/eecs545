import numpy as np
from numpy.linalg import inv
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50;
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

# 90% Training set, 10% Validation set
Nsplit_new = round(len(X_train)*0.1);
X_train_new, y_train_new = X_train[:-Nsplit_new], y_train[:-Nsplit_new]
X_train_val, y_train_val = X_train[-Nsplit_new:], y_train[-Nsplit_new:];

# Closed form solution
def cfs(x_train, y_train, lmda):
    N = len(x_train);
    I = np.identity(len(x_train[0])+1);
    attri = np.c_[np.ones(len(x_train)), x_train];
    coef = np.dot(inv(np.dot(attri.T, attri) + N*lmda*I), np.dot(attri.T,y_train));
    loss = 1/N * (1/2*coef.T.dot(attri.T).dot(attri).dot(coef) - y_train.T.dot(attri).dot(coef) + 1/2 * y_train.T.dot(y_train));
    return coef, loss;

# Loss function
def lossFn(x_test, y_test, lmda, coef):
    N = len(x_test);
    attri = np.c_[np.ones(len(x_test)), x_test];
    loss = 1/N * (1/2*coef.T.dot(attri.T).dot(attri).dot(coef) - y_test.T.dot(attri).dot(coef) + 1/2 * y_test.T.dot(y_test));
    return loss;

# Root mean-squared error
def rmse(coef, soln, actual):
    featureVector = np.c_[np.ones(len(soln)), soln];
    return np.sqrt(np.mean((np.dot(featureVector, coef) - actual)**2));

# Main function
if __name__ == "__main__":
    lmdas = np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5]);
    i = 0;
    coef  = np.zeros((len(X_train[0])+1, len(lmdas)));
    loss  = np.zeros((1, len(lmdas)));
    error_val = np.zeros((1, len(lmdas)));
    error_tst = np.zeros((1, len(lmdas)));
    error_tra = np.zeros((1, len(lmdas)));
    
    for lmda in lmdas:
        coef[:,i], _ = cfs(X_train_new, y_train_new, lmda);
        loss[:,i] = lossFn(X_train_val, y_train_val, lmda, coef[:,i].T)
        # Validation Error
        error_val[:,i] = rmse(coef[:,i], X_train_val, y_train_val);
        # Testing Error
        error_tst[:,i] = rmse(coef[:,i], X_test, y_test);
        # Training Error
        error_tra[:,i] = rmse(coef[:,i], X_train_new, y_train_new);
        i += 1;
    
    # Estimate the loss on the validation set for each of the regularization parameter
    plt.figure()
    plt.plot(lmdas, loss.T);
    plt.grid(True);
    plt.xlabel("Lambda");
    plt.ylabel("Validation Loss Function");
    # Report the lowest RMSE error identified
    plt.figure()
    plt.plot(lmdas, error_tst.T, label = "Test");
    plt.plot(lmdas, error_val.T, label = "Validation");
    plt.plot(lmdas, error_tra.T, label = "Training");
    plt.grid(True);
    plt.xlabel("Lambda");
    plt.ylabel("Error");
    plt.legend()
    




























