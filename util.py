import os
import random
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):

    dataMat = []
    lableMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        lableMat.append(float(lineArr[2]))
    return dataMat,lableMat

def selectJrand(i,m):

    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels

        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]

        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.b = 0


def calcEk(os, k):
    fXk = float(np.multiply(os.alphas, os.labelMat).T * os.X * os.X[k, :].T) + os.b
    Ek = fXk - os.labelMat[k, :]
    return Ek


def selectJ(i, os, Ei):
    selectj = -1
    maxDeE = 0
    Ej = 0

    os.eCache[i] = [1, Ei]
    # mat.A : convert type mat to array, eg. a = mat([[0]]) b = a.A ,b's type is array
    validEcacheList = np.nonzero(os.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(os, k)
            DeE = abs(Ei - Ek)
            if DeE > maxDeE:
                maxDeE = DeE
                selectj = k
                Ej = Ek
        return selectj, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calcEk(os, j)
        return j, Ej


def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]



def calcWs(alphas, dataArr,classLables):

    X = np.mat(dataArr);labelMat = np.mat(classLables).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def plot_line(dataMat,labelMat,alphas,w,b):

    #print dataArr[:,0]
    #Point
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(dataMat)):
        if labelMat[i] > 0 :
            ax.scatter(dataMat[i,0].transpose(),dataMat[i,1].transpose(), marker='s', s=30)
        else:
            ax.scatter(dataMat[i, 0].transpose(), dataMat[i, 1].transpose(), marker='o',s=50, c='red')
    #Line
    x1 = max(dataMat[:,0])
    x2 = min(dataMat[:,0])
    a1,a2 = w
    a1 = float(a1[0]);a2 = float(a2[0]);b = float(b)
    x = np.arange(x2,12.0,x1)
    y = (-a1*x-b)/a2
    ax.plot(x,y)

    #Support Vector
    for i,alpha in enumerate(alphas):
        if alpha > 0:
            x,y = dataMat[i,0],dataMat[i,1]
            ax.scatter(x,y,s = 150, c='none',alpha=0.7, linewidth=1.5, edgecolor='red')

    plt.show()


if __name__ == '__main__':
    import numpy as np
    data,lable = loadDataSet('Ch06\\testSet.txt')
    lable = np.mat(lable).transpose()
    print(lable.shape)