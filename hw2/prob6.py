import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Normalized Binary dataset
# 4 features, 100 examples, 50 labeled 0 and 50 labeled 1
X, y = load_breast_cancer().data, load_breast_cancer().target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

tol = 1e-6;

# Sigmoid function
def sigmoid(a):
    return np.exp(a)/(1+np.exp(a));

# Cross-entropy Loss Function
def crossEntropyLoss(w, x, t):
    return -(np.dot(t, np.log(sigmoid(np.dot(x,w)))) + np.dot((1-t), np.log(1 - sigmoid(np.dot(x,w)))));

# Average cross-entropy Loss 
def averageCrossEntropyLoss(w, x, t):
    return crossEntropyLoss(w, x, t)/len(t);

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

# Computing classification accuracy
def accuracy(w, x, t):
    t_predict = (np.dot(w, x.T) >= 0).astype(np.int);
    correctPrediction = np.sum(t_predict==t);
    return correctPrediction/len(t);

# Stochastic Gradient Descent
def sgd(test, train, learn_rate, n_epoch):
    # Initialize the weight vector randomly from a uniform distribution
    w = [ np.random.uniform(low = -1, high = 1, size=None) for i in range(len(train[0]) - 1) ];
    itr = 0;
    error_train, error_test = [], [];
    acc_train, acc_test = [], [];
    for epoch in range(n_epoch):
        # Random Shuffle
        np.random.shuffle(train);
        # Extracting information and adding the bias term
        X_train,  t_train = train[:,0:-1], train[:,-1]; 
        X_test,    t_test =  test[:,0:-1],  test[:,-1];
        for rowx, rowt in zip(X_train, t_train):
            grad = rowx.T.dot(sigmoid(rowx.dot(w)) - rowt);
            w -= learn_rate*grad;
            itr += 1;
            error_train.append(averageCrossEntropyLoss(w, X_train, t_train));
            error_test.append(averageCrossEntropyLoss(w, X_test, t_test));
            acc_train.append(accuracy(w, X_train, t_train));
            acc_test.append(accuracy(w, X_test, t_test));
        print('> epoch=%.0f, LearningRate=%.5f, Error=%.6f' % (epoch+1, learn_rate, error_train[-1]));
        # Convergence critera: break when training error does not change much
        if epoch > 0 and (np.abs(error_train[-1] - error_train[-2]) < tol):
            break;
    return w, error_train, error_test, acc_train, acc_test, itr;


if __name__ == "__main__":
    X_train_scaled = featureNormalization(X_train);
    X_test_scaled  = featureNormalization(X_test);
    XY_train = np.c_[X_train_scaled, y_train];
    XY_test  = np.c_[ X_test_scaled,  y_test];
    
    w, error_train, error_test, acc_train, acc_test, itr = sgd(XY_test, XY_train, 1e-2, 1);
    
    plt.figure()
    plt.plot(range(itr), acc_train, label = "Training");
    plt.plot(range(itr),  acc_test, label = "Test");
    plt.xlabel('SGD Iteration');
    plt.ylabel('Classification Accuracy');
    plt.grid(True);
    plt.legend()
    
    plt.figure()
    plt.plot(range(itr), error_train, label = "Training");
    plt.plot(range(itr),  error_test, label = "Test");
    plt.xlabel('SGD Iteration');
    plt.ylabel('Average Error');
    plt.grid(True);
    plt.legend()



























