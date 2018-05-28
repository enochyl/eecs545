import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

np.random.seed(3)

mean_1 = [ 2.0 , 0.2 ]
cov_1 = [ [ 1 , .5 ] , [ .5 , 2.0 ]]

mean_2 = [ 0.4 , -2.0 ]
cov_2 = [ [ 1.25 , -0.2 ] , [ -0.2, 1.75 ] ]

x_1 , y_1 = np.random.multivariate_normal( mean_1 , cov_1, 15).T
x_2 , y_2 = np.random.multivariate_normal( mean_2 , cov_2, 15).T

X = np.zeros((30,2))
X[0:15,0] = x_1
X[0:15,1] = y_1
X[15:,0] = x_2
X[15:,1] = y_2

y = np.zeros(30)
y[0:15] = np.ones(15)
y[15:] = -1 * np.ones(15)

def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out

if __name__ == "__main__":
    Cs_1a = [1.0, 100.0];
    Cs_1b = [1.0, 3.0];
    xAxis = np.linspace(-4, 6, 1000);
    yAxis = np.linspace(-6, 3, 1000);
    xx, yy = np.meshgrid(xAxis, yAxis);
    # fit the model
    for C in Cs_1a:
        clf = svm.SVC(kernel='linear', C = C);
        clf.fit(X, y);
        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        yhyperplane = a * xAxis - (clf.intercept_[0]) / w[1]
        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        b = clf.support_vectors_[0]
        yy_down = a * xAxis + (b[1] - a * b[0])
        b = clf.support_vectors_[-1]
        yy_up = a * xAxis + (b[1] - a * b[0])
        plt.figure();
        plt.plot(xAxis, yhyperplane, 'k-')
        # plt.plot(xAxis, yy_down, 'k--')
        # plt.plot(xAxis, yy_up, 'k--')
        # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
        #     s=80, facecolors='none')
        plt.plot( x_1 , y_1 , 'x' )
        plt.plot( x_2 , y_2 , 'ro')
        plt.title('C = %i' %C);
        plt.show();
        print(clf.n_support_);


    for C in Cs_1b:
        clf = svm.SVC(kernel='rbf', C = C);
        clf.fit(X, y);
        plt.figure();
        plot_contours(clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8);
        plt.plot( x_1 , y_1 , 'x' )
        plt.plot( x_2 , y_2 , 'ro')
        plt.title('C = %i' %C);
        plt.show();

        n_SV = clf.n_support_;
        print(n_SV);
