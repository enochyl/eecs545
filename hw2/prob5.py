import numpy as np
import matplotlib.pyplot as plt


tol = 1e-6;
data = np.zeros((100, 3))
val = np.random.uniform(0, 2, 100)
diff = np.random.uniform(-1, 1, 100)
data[:,0], data[:,1], data[:,2] = val - diff, val + diff, np.ones(100)
target = np.asarray(val > 1, dtype = int) * 2 - 1


datab = np.ones((100, 3))
datab[:50,0], datab[50:,0] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
datab[:50,1], datab[50:,1] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
targetb = np.zeros(100)
targetb[:50], targetb[50:] = -1 * np.ones(50), np.ones(50)

datab_original = datab.copy();

data  = np.c_[data, target];

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


def perceptron(train, n_epoch, mode):
    # Weight vector initialization
    coef = [0.0 for i in range(len(train[0])-1)];
    if mode == 0:
        for epoch in range(n_epoch):
            for row in train:
                if row[-1]*(np.dot(row[0:-1],coef)) > 0:
                    continue;
                else:
                    coef += row[-1]*row[0:-1];
    elif mode == 1:
        for epoch in range(n_epoch):
            for row in train:
                if row[-1]*(np.dot(row[0:-1],coef)) > 0:
                    continue;
                else:
                    coef += 0.05*row[-1]*row[0:-1];
    return coef;

if __name__ == "__main__":
    w = perceptron(data, 10, 0);
    # Reconstruct classification line
    t = np.linspace(min(data[:,1]),max(data[:,1]),101);
    y = (-w[2] - w[0]*t)/w[1];
    
    plt.figure();
    plt.scatter(data[:,0], data[:,1], c = target);
    plt.plot(t,y);
    plt.xlabel('$x$');
    plt.ylabel('$y$');
    plt.grid(True);
    

    datab = featureNormalization(datab);
    datab = np.c_[datab, targetb];
    t = np.linspace(min(datab[:,1]),max(datab[:,1]),101);
    wb = perceptron(datab, 10000, 1);
    yb = (-wb[2] - wb[0]*t)/wb[1];
    plt.figure();
    plt.scatter(datab[:,0], datab[:,1], c = target);
    plt.plot(t,yb);
    plt.xlabel('$x$');
    plt.ylabel('$y$');
    plt.grid(True);
    
    
    
    
    