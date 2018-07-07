from util import *
from SMOSimple import *
from SMOPlatt import *
from numpy import *

if __name__ == '__main__':
    dataArr,labelArr = loadDataSet('E:\\SVM\\Ch06\\testSet.txt')
    b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
    w = calcWs(alphas,dataArr,labelArr)

    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    alphas = mat(alphas)

    plot_line(dataMat,labelMat,alphas,w,b)