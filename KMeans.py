import numpy as np
import os
import sys
import operator
import matplotlib
import matplotlib.pyplot as plt
path =  sys.path[0]+"\data"
'''
现在一般都用panda了但是还是学一下怎么直接读txt
'''

def file2matix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    numberOfCols =len(arrayOLines[0].split('\t'))-1
    returnMat = np.zeros((numberOfLines-10,numberOfCols))
    classLabelVector = []
    index = 0
    #最后十行用来测试
    for line in arrayOLines[0:-10]:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:numberOfCols]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    Testdata = []
    for line in arrayOLines[numberOfLines-10:]:
        line = line.strip()
        listFromLine = line.split('\t')
        Testdata.append(listFromLine[0:numberOfCols])

    return returnMat,classLabelVector,Testdata
'''
k邻近算法
inx是输入要判断的数据 dataset数据集(nparray) labels是标签 k是维数
'''
def classify0(inX,dataset,labels,k):
    dataSetSize = dataset.shape[0]
    diffMat  = np.tile(inX,(dataSetSize,1))-dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    #对大小进行标签排序
    sortedDisIndicies = distances.argsort()
    classCount= {}
    for i in range(k):
        votelLabel = labels[sortedDisIndicies[i]]
        #如果是相同类就加一，如果字典的键值代表次数，需要不断累加就这么写
        classCount[votelLabel] = classCount.get(votelLabel,0)+1
    '''
    sort 与 sorted 区别：

sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。

list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
    
    '''
    #按照字典的值进行降序排列，返回值是一个列表，列表的元素是元组
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),
                reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

def DataVisualization(RetM,classL,Test):
    #numpy拼接操作有毒，怎么写的，还要用python的列表？以后还是用panda吧
    ListRetM = RetM.tolist()
    for index,value in enumerate(ListRetM):
        tmp  = ListRetM[index]
        tmp.append(classL[index])
        ListRetM[index] = tmp
    fig = plt.figure()
    ax = fig.add_subplot(111)
    npdatemat = np.array(ListRetM)
    #这样写看不出来类别
    # ax.scatter(npdatemat[:,1],npdatemat[:,2])
    #这样写可以看出来类别
    ax.scatter(npdatemat[:, 1], npdatemat[:, 2],
    15.0*np.array(classL),15*np.array(classL))

    ax.set_xlabel('time of playing video game')
    ax.set_ylabel('consumption of ice-cream')
    plt.show()

if __name__=='__main__':
    #返回类型
    # RetM <class 'numpy.ndarray'>
    # classL <class 'list'>
    #Test <class 'list'>
    RetM,classL,Test = file2matix(path+'\\'+'datingTestSet2.txt')
    for each in Test:
        inX = [float(i) for i in each]
        cator = classify0(inX,RetM,classL,10)
        print(cator)
    #从结果来看，准确率80%，后两个错了
    DataVisualization(RetM,classL,Test)

