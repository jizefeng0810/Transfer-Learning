import numpy as np
import time
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                src, tar = './data/' + domains[i], './data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                clf = KNeighborsClassifier(n_neighbors=1)  # create k neighbors classifier
                clf.fit(Xs, Ys.ravel())  # classify
                Y_tar_pseudo = clf.predict(Xt)  # predict pseudo labels
                acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)  # calculate score
                print(domains[i]+'--'+domains[j]+': '+str(acc))



