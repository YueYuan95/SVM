import numpy as np

from util import *

def innerL(i,os):
    Ei = calcEk(os,i)
    #I don't get it
    if ((os.labelMat[i]*Ei < -os.tol) and (os.alphas[i] < os.C)) or ((os.labelMat[i]*Ei > os.tol) and (os.alphas[i] > 0)):
        #
        j,Ej = selectJ(i,os,Ei)
        #
        alphasIold = os.alphas[i].copy();alphasJold = os.alphas[j].copy()

        #calculate borad
        if (os.labelMat[i] != os.labelMat[j]):
            L = max(0,os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0,os.alphas[j]+os.alphas[i]-os.C)
            H = min(os.C ,os.alphas[j]+os.alphas[i])
        if L==H: print "L==H";return 0
        #
        eta = 2.0*os.X[i,:]*os.X[j,:].T - os.X[i,:]*os.X[i,:].T - os.X[i,:]*os.X[i,:].T
        if eta >= 0: print "eta >= 0";return 0
        #
        os.alphas[j] -= os.labelMat[j]*(Ei-Ej)/eta
        os.alphas[j] = clipAlpha(os.alphas[j],H,L)

        if (abs(os.alphas[j]-alphasJold) < 0.00001): print "j is not move enough"; return 0

        os.alphas[i] = alphasIold + os.labelMat[i]*os.labelMat[j]*(alphasJold-os.alphas[j])

        b1 = os.b - Ei - os.labelMat[i]*os.X[i,:]*os.X[i,:].T*(os.alphas[i] - alphasIold) - \
            os.labelMat[j]*os.X[j,:]*os.X[i,:].T*(os.alphas[j] - alphasJold)

        b2 = os.b - Ej - os.labelMat[i]*os.X[i,:]*os.X[j,:].T*(os.alphas[i] - alphasIold) - \
            os.labelMat[j]*os.X[j,:]*os.X[j,:].T*(os.alphas[j] - alphasJold)

        if ((os.alphas[i]>0) and (os.alphas[i]<os.C)): os.b = b1
        elif ((os.alphas[j]>0) and (os.alphas[j]<os.C)): os.b = b2
        else: os.b = (b1 + b2)/2.0

        updateEk(os,i)
        updateEk(os,j)

        return 1
    else:
        return 0


def smoP(dataMatIn,classLabels,C,toler,maxIter,KTup=('lin',0)):

    os = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True;alphaPairChanged = 0

    while ((iter < maxIter) and ((alphaPairChanged>0) or (entireSet))):
        alphaPairChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairChanged += innerL(i,os)
            print "fullSet, iter: %d , i: %d, pairs changed: %d" % (iter , i ,alphaPairChanged)
            iter += 1
        else:

            supportVector = np.nonzero(((os.alphas.A >0)*(os.alphas.A<os.C)))[0]
            for i in supportVector:
                alphaPairChanged += innerL(i,os)
            print "Support Vector, iter: %d , i:%d, pairs changed: %d" % (iter, i, alphaPairChanged)
            iter += 1

        if entireSet: entireSet = False
        elif (alphaPairChanged == 0): entireSet = True

        print "iteration number: %d " % iter

    return os.b, os.alphas


