import numpy as np

from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC

mnist = fetch_mldata('MNIST original', data_home='./')

#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target

N = len(images)
np.random.seed(1234)
inds = np.random.permutation(N)
images = np.array([images[i] for i in inds])
targets = np.array([targets[i] for i in inds])

# Normalize data
X_data = images/255.0
Y = targets

# Train/test split
X_train, y_train = X_data[:10000], Y[:10000]
X_test, y_test = X_data[-10000:], Y[-10000:]

clf = svm.SVC(kernel='rbf', C = 1, gamma = 1);
clf.fit(X_train, y_train);
y_predict = clf.predict(X_test);
accuracy = np.sum(y_predict == y_test)/len(y_test);
print(accuracy);
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [5e-2, 1e-1, 5e-1, 1.0],
                     'C': [1, 3, 5]}];
clf = GridSearchCV(SVC(), tuned_parameters, cv = 5, n_jobs = 4);
clf.fit(X_train, y_train);

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
