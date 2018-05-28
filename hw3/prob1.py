import numpy as np
import pickle

# load the training data 
train_features = np.load("spam_train_features.npy")
train_labels = np.load("spam_train_labels.npy")


# load the test data 
test_features = np.load("spam_test_features.npy")
test_labels = np.load("spam_test_labels.npy")


def getPi(labels, N, c):
    Nc = sum(labels == c);
    return (Nc + 1)/(N + 2);
    
def getTheta(features, labels, c, d, m):
    Nc = sum(labels == c);
    train = np.c_[features, labels];
    Ncdm = sum((train[:,-1] == c) & (train[:,d] == m));
    theta_cdm = (Ncdm + 1)/(Nc + 2);
    return theta_cdm;

def computeTheta(features, labels, C, D, M):
    theta_cdm = [];
    # Create an empty 3-dimensional array for theta
    for c in range(C):
        theta = [];
        for d in range(D):
            column = [];
            for m in range(M):
                column.append(0);
            theta.append(column);
        theta_cdm.append(theta);
    # Populate theta_cdm
    for c in range(C):
        for d in range(D):
            for m in range(M):
                theta_cdm[c][d][m] = getTheta(features, labels, c, d, m);
    return theta_cdm;
    
def getLikelihood(features, labels, N, C, D, M, x, pi, theta_cdm):
    for n in range(N):
        for c in range(C):
            front = (labels[n] == c)*np.log(pi);
            for d in range(D):
                for m in range(M):
                    end = (features[n,d] == m)*(labels[n] == c)*np.log(theta_cdm);  
    return front + end;

def classifyNaiveBayes(D, M, test_features, c, theta_, pi):
    temp = 1;
    for d in range(D):
        for m in range(M):
            temp *= (theta_[c][d][m])**(m == test_features[d]);
    return pi*temp;
            
if __name__ == "__main__":
    # Constants Definition
    N = len(train_features);
    D = len(train_features[0]);
    C = len(np.unique(train_labels));
    M = len(np.unique(train_features));
    
    Pi0 = getPi(train_labels, N, 0);
    Pi1 = getPi(train_labels, N, 1);
    
    theta_cdm = computeTheta(train_features, train_labels, C, D, M);
    with open("theta_cdm.txt", "wb") as fp:
        pickle.dump(theta_cdm, fp);
    with open("theta_cdm.txt", "rb") as fp:   # Unpickling
        theta_cdm = pickle.load(fp);
    
    predicted_labels = [];
    for n in range(len(test_features)):
        temp_class0 = classifyNaiveBayes(D, M, test_features[n,:], 0, theta_cdm, Pi0);
        temp_class1 = classifyNaiveBayes(D, M, test_features[n,:], 1, theta_cdm, Pi1);
        if temp_class0 > temp_class1:
            predicted_labels.append(0);
        else:
            predicted_labels.append(1);
    prediction_difference = np.abs(test_labels - predicted_labels);
    errorRate = sum(prediction_difference == 0)/len(prediction_difference)*100;
    print("Error Rate = ", errorRate,"%");




















    
    