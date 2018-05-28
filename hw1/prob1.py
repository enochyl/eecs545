import numpy as np
from numpy.linalg import inv
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target
# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []

#Nsplit = 50
## Training set
#X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
## Test set
#X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

# Normalizing features
X_train_scaled = np.zeros((len(X_train),len(X_train[0])));
X_test_scaled  = np.zeros((len(X_test), len( X_test[0])));
for i in range(len(X_train[0])):
    stdev = np.std(X_train[:,i]);
    mean  = np.mean(X_train[:,i]);
    if np.abs(stdev) < 1e-8:
        X_train_scaled[:,i] = (X_train[:,i] - mean);
    else:
        X_train_scaled[:,i] = (X_train[:,i] - mean)/(stdev);
 
for i in range(len(X_test[0])):
    stdev = np.std(X_test[:,i]);
    mean  =  np.mean(X_test[:,i]);
    if np.abs(stdev) < 1e-8:
        X_test_scaled[:,i] = (X_test[:,i] - mean);
    else:
        X_test_scaled[:,i] = (X_test[:,i] - mean)/(stdev);

# Error tolerance
tol = 1e-6;

# Appending target values to attributes
XY_train = np.c_[X_train_scaled, y_train];
XY_test  = np.c_[ X_test_scaled,  y_test];

def preprocess(X_train):
    X_train_scaled = np.zeros((len(X_train),len(X_train[0])));
    for i in range(len(X_train[0])):
        stdev = np.std(X_train[:,i]);
        mean  = np.mean(X_train[:,i]);
        if np.abs(stdev) < 1e-8:
            X_train_scaled[:,i] = (X_train[:,i] - mean);
        else:
            X_train_scaled[:,i] = (X_train[:,i] - mean)/(stdev);
    return np.c_[np.ones(len(X_train)), X_train_scaled];
        
# Closed form solution (1e)
def solve(x_train, y_train):
    coef = inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train);
    return coef

# Mean-squared Error
def mse(X, Y, w):
    return np.sum((np.dot(X, w) - Y)**2)/len(X);

# Make prediction based on input features and trained coefficients
def linearPredict(row, coefficients):
    # bias term = 1, not responsible for a specific input value
    yhat = coefficients[0];
    for i in range(len(row)-1):
        yhat += coefficients[i+1] * row[i];
    return yhat;

# Stochastic gradient descent
def sgd(test, train, learn_rate, n_epoch):
    # Initialize the weight vector randomly from a uniform distribution
    coef = [ np.random.uniform(low = -0.1, high = 0.1, size=None) for i in range(len(train[0])) ];
    train_error = []; test_error = [];
    for epoch in range(n_epoch):
        # Random Shuffle
        np.random.shuffle(train);
        # Extracting information and adding the bias term
        attri       = np.c_[np.ones(len(X_train)), train[:,0:-1]]; 
        attri_test  = np.c_[np.ones(len(X_test)), test[:,0:-1]];
        target      = train[:,-1];
        target_test = test[:,-1];
        for row in train:
            yhat  = linearPredict(row, coef);
            error = yhat - row[-1];
            # Bias term gradient = 1
            coef[0] = coef[0] - learn_rate*error;
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1] - learn_rate*error*row[i]
        train_error.append(np.sum(np.square(np.dot(attri, coef) - target))/len(attri));
        test_error.append(np.sum(np.square(np.dot(attri_test, coef) - target_test))/len(attri_test));
        print('> epoch=%.0f, LearningRate=%.5f, Error=%.6f' % (epoch+1, learn_rate, train_error[epoch]));
        # Convergence critera: break when training error does not change much
        if epoch > 0 and (np.abs(train_error[epoch] - train_error[epoch-1]) < tol):
            break;
    return np.asarray(coef), epoch, train_error, test_error;

# Batch gradient descent
def bgd(test, train, learn_rate, n_epoch):
    # Initialize the weight vector
    coef = [0.0 for i in range(len(train[0])) ];
    train_error = []; test_error = [];
    attri       = np.c_[np.ones(len(X_train)), train[:,0:-1]]; 
    attri_test  = np.c_[ np.ones(len(X_test)), test[:,0:-1]];
    target      = train[:,-1];
    target_test = test[:,-1];
    for epoch in range(n_epoch):
        yhat  = attri.dot(coef);
        error = yhat - target;
        grad  = attri.T.dot(error);
        coef  = coef - learn_rate*grad;
        train_error.append(np.sum(np.square(np.dot(attri, coef) - target))/len(attri));
        test_error.append(np.sum(np.square(np.dot(attri_test, coef) - target_test))/len(attri_test));
        print('> epoch=%.0f, LearningRate=%.5f, Error=%.6f' % (epoch+1, learn_rate, train_error[epoch]));
        # Convergence critera: break when training error does not change much
        if epoch > 0 and (np.abs(train_error[epoch] - train_error[epoch-1]) < tol):
            break;
    return coef, epoch, train_error, test_error;


# Closed form solution
def cfs(x_train, y_train):
    attri = np.c_[x_train, np.ones(len(x_train))];
    coef = inv(attri.T.dot(attri)).dot(attri.T).dot(y_train);
    return coef
    
# Main function
if __name__ == "__main__":   
#    coef_s, epoch_s, train_error_s, test_error_s = sgd(XY_test, XY_train, 1e-6, 2000);
#    coef_b, epoch_b, train_error_b, test_error_b = bgd(XY_test, XY_train, 1e-6, 2000);
    coef_c = cfs(X_train_scaled, y_train);
    train_error_c = mse(np.c_[X_train_scaled, np.ones(len(X_train_scaled))], y_train, coef_c);
    test_error_c  = mse(np.c_[X_test_scaled, np.ones(len(X_test_scaled))], y_test, coef_c);
    

    plt.figure()
    plt.plot(range(epoch_s+1), train_error_s, label = "SGD");
    plt.grid(True);
    plt.xlabel("Epoch");
    plt.ylabel("Error");
    plt.legend()
    
    plt.figure()
    plt.plot(range(epoch_b+1), train_error_b, label = "BGD");
    plt.grid(True);
    plt.xlabel("Epoch");
    plt.ylabel("Error");
    plt.legend()
    
    plt.figure()
    plt.plot(range(epoch_s+1), train_error_s, label = "Training error (SGD)");
    plt.plot(range(epoch_s+1), test_error_s, label = "Testing error (SGD)");
    plt.grid(True)
    plt.xlabel("Epoch");
    plt.ylabel("Error");
    plt.legend()
    
    plt.figure()
    plt.plot(range(epoch_b+1), train_error_b, label = "Training error (BGD)");
    plt.plot(range(epoch_b+1), test_error_b, label = "Testing error (BGD)");
    plt.grid(True)
    plt.xlabel("Epoch");
    plt.ylabel("Error");
    plt.legend()
    
    for k in range(100):

  # Shuffle data
        rand_perm = np.random.permutation(Ndata)
        features = [features_orig[ind] for ind in rand_perm]
        labels = [labels_orig[ind] for ind in rand_perm]

  # Train/test split
        Nsplit = 50
        X_train_e, y_train_e = features[:-Nsplit], labels[:-Nsplit]
        X_test_e, y_test_e = features[-Nsplit:], labels[-Nsplit:]

        X_train_e = preprocess(np.asarray(X_train_e));
        X_test_e = preprocess(np.asarray(X_test_e));

        # Solve for optimal w
        # Use your solver function
        w = solve(X_train_e, y_train_e)

    # Collect train and test errors
    # Use your implementation of the mse function
        train_errs.append(mse(X_train_e, y_train_e, w))
        test_errs.append(mse(X_test_e, y_test_e, w))

    print('Mean training error: ', np.mean(train_errs))
    print('Mean test error: ', np.mean(test_errs))

                
                
                
                
                
                
            #Find the bug    
                
                
                
                
                
                
                
                
                
        