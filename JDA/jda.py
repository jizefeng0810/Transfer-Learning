# encoding=utf-8
"""
    Created on 21:29 2019/10/23
    @author: Zefeng Ji
"""
import numpy as np
import time
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        iteration = []
        X = np.hstack((Xs.T, Xt.T))     # cat horizontally
        X /= np.linalg.norm(X, axis=0)  # normalization
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1)))) # cat vertically and normalization
        C = len(np.unique(Ys))  # class numbers
        H = np.eye(n) - 1 / n * np.ones((n, n)) # central matrix

        M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            starttime = time.time()     # start time
            N = 0
            M0 = e * e.T * C    # MMD matrix
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N  # update M
            M = M / np.linalg.norm(M, 'fro')                            # M normalization
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)     # kernel function
            n_eye = m if self.kernel_type == 'primal' else n            #
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])   # （X*M*X.T+lamb*I) = (X*H*X.T)
            w, V = scipy.linalg.eig(a, b)   # Find eigenvalues w and eigenvector V
            ind = np.argsort(w)     # eigen value sort
            A = V[:, ind[:self.dim]]            # fetch dim eigen vecter
            Z = np.dot(A.T, K)                  # Z = A.T*K
            Z /= np.linalg.norm(Z, axis=0)      # Z normalization
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)   # create k neighbors classifier
            clf.fit(Xs_new, Ys.ravel())                 # classify
            Y_tar_pseudo = clf.predict(Xt_new)          # predict pseudo labels
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)  # calculate score
            endtime = time.time()               # end time
            cost_time = endtime - starttime     # cost time
            list_acc.append(acc)
            iteration.append(t+1)
            print('JDA iteration [{}/{}]: Acc: {:.4f}: CostTime:{:.4f}s'.format(t + 1, self.T, acc, cost_time))
        return acc, Y_tar_pseudo, list_acc,iteration


if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(0,4):
        for j in range(0,4):
            if i != j:
                src, tar = './data/' + domains[i], './data/' + domains[j]
                print(src,tar)
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                jda = JDA(kernel_type='primal', dim=50, lamb=1, gamma=1, T=10)
                acc, ypre, list_acc, iteration = jda.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(iteration,list_acc)
                plt.show()