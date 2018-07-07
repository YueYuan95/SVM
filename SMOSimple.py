import util
import numpy as np

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):

    dataMatrix = np.mat(dataMatIn);labelMatrix = np.mat(classLabels).transpose()
    m,n = dataMatrix.shape
    b = 0
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while(iter<maxIter):
        alphasPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMatrix).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMatrix[i])
            if ((labelMatrix[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i]*Ei > toler and alphas[0]>0)):
                j = util.selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMatrix).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMatrix[j])
                alphasIold = alphas[i].copy();alphasJold = alphas[j].copy()
                if (labelMatrix[i] != labelMatrix[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L==H: print('L is equal to H');continue
                eta = dataMatrix[i,:]*dataMatrix[i,:].T +dataMatrix[j,:]*dataMatrix[j,:].T - 2.0*dataMatrix[i,:]*dataMatrix[j,:].T
                if eta < 0 : print "eta < 0";continue
                alphas[j] += labelMatrix[j]*(Ei - Ej)/eta
                alphas[j]  = util.clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphasJold) < 0.00001):
                    print "j not moving enough";continue
                alphas[i] += labelMatrix[i]*labelMatrix[j]*(alphasJold-alphas[j])
                b1 = b - Ei - labelMatrix[i]*dataMatrix[i,:]*dataMatrix[i,:].T*(alphas[i]-alphasIold)- \
                    labelMatrix[j]*dataMatrix[j,:]*dataMatrix[i,:].T*(alphas[j]-alphasJold)
                b2 = b - Ej - labelMatrix[i]*dataMatrix[i,:]*dataMatrix[j,:].T*(alphas[i]-alphasIold)- \
                    labelMatrix[j]*dataMatrix[j,:]*dataMatrix[j,:].T*(alphas[j]-alphasJold)
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif(0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1+b2)/2
                alphasPairsChanged += 1
                print "iter:%d  i:%d, pairs changed %d" % (iter, i, alphasPairsChanged)
        if (alphasPairsChanged == 0 ): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter

    return b,alphas



