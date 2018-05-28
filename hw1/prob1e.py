import numpy as np
from numpy.linalg import inv
from sklearn import datasets
import matplotlib.pyplot as plt

# Load dataset
dataset = datasets.load_boston()

# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []


    
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
        
# Closed form solution
def solve(x_train, y_train):
    coef = inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train);
    return coef

# Mean-squared Error
def mse(X, Y, w):
    return np.mean((np.dot(X, w) - Y)**2);

for k in range(100):

  # Shuffle data
  rand_perm = np.random.permutation(Ndata)
  features = [features_orig[ind] for ind in rand_perm]
  labels = [labels_orig[ind] for ind in rand_perm]

  # Train/test split
  Nsplit = 50
  X_train_e, y_train_e = features[:-Nsplit], labels[:-Nsplit]
  X_test_e, y_test_e = features[-Nsplit:], labels[-Nsplit:]

  # Preprocess your data - Normalization, adding a constant feature
#  params = preproc_params(X_train)
#  X_train = preprocess(X_train, params)
#  X_test = preprocess(X_test, params)

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
