import pandas as pd
import numpy as np
import sys
from math import log
import operator
from sklearn.tree import DecisionTreeClassifier
path =  sys.path[0]+"\data\\"
def Load_data(name):
    data = pd.read_csv(path+name)
    print(data.head())
    rows = len(data)
    print(rows)#多少行
    One_Shot_Data = pd.DataFrame()
    listTags =  list(data["class"])
    for i,j in enumerate(listTags):
        if listTags[i]=='e':
            listTags[i] = 1
        else:
            listTags[i] = 0
    Tags = np.array(listTags)
    del data["class"]
    for cols in (data.columns):
        # if count>0:
        #     break
        # count+=1
        #有多少种
        CatorNum = np.unique(data[cols])
        Ncols = len(CatorNum)
        IndexDict = {}
        for  index,tags in enumerate(CatorNum):
            IndexDict[tags] = index
        One_Shot_Feature = np.zeros((rows,Ncols))

        for index,rowdata in enumerate(One_Shot_Feature):
            feature = data[cols][index]
            rowdata[IndexDict[feature]] = 1

        for i in range(Ncols):
            featurename = cols + str(i)
            tmp = One_Shot_Feature[:,i]
            # One_Shot_Data[featurename] = [int(i) for i in tmp ]
            One_Shot_Data[featurename] = tmp
    # One_Shot_Data.to_csv("test.csv",index=False)

    return One_Shot_Data,Tags
def SplitDataSet(data,axis,value):
    retdata = []
    for featVec in data:
        if featVec[axis]==value:
            reducedFeaVect = featVec[:axis]
            list(reducedFeaVect).extend( featVec[axis+1:])
            retdata.append(list(reducedFeaVect))
    return  retdata
#注意这个dataset是没有去掉label的data
def calshannonEnt(Tags):
    numentry = len(Tags)
    labelcount = {}
    for feaVet in Tags:
        if feaVet not in labelcount.keys():
            labelcount[feaVet] = 0
        labelcount[feaVet]+=1
    shannonEnt = 0.0
    for key in labelcount:
        prob = float(labelcount[key])/numentry
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
#这个dataset没有去掉特征向量之前的
def chooseBestFeatureToSplit(dataset,Tags):
     dataset = np.array(dataset)
     baseEnt = calshannonEnt(dataset[:,0])
     bestInfoGain = 0.0
     bestFeature = -1

     for cols in range(1,len(dataset[0])):
         featlist = dataset[:,cols]
         uniqueVals  = set(featlist)
         newEnt = 0.0
         for v in uniqueVals:
             subdataset = SplitDataSet(dataset.tolist(),cols,v)
             prob = len(subdataset)/float(len(dataset))
             #把标签切割出来
             npsubdata  = np.array(subdataset)
             subTags = npsubdata[:,0]
             newEnt+=prob*calshannonEnt(subTags)
             infoGain = baseEnt - newEnt
         print("第%d个特征的增益为%.3f"%(cols,infoGain))
         if(infoGain>bestInfoGain):
             bestInfoGain  = infoGain
             bestFeature = cols
     print("第%d个特征的增益最大，为%.3f" % (bestFeature,bestInfoGain))
     return bestFeature
def majorityCnt(classlist):
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys(): classcount[vote] = 0
        classcount[vote]+=1
    sortedclasscount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return  sortedclasscount[0][0]
def CreateTree(dataset,labels,featlabels):
    labels = list(labels)
    npdata = np.array(dataset)
    classlist  = [example[0] for example in npdata]
    if classlist.count(classlist[0])==len(classlist):#如果类别完全相同
        return classlist[0]
    if len(npdata[0])==1:
        return majorityCnt(classlist)
    bestfeat = chooseBestFeatureToSplit(dataset,labels)

    bestfeatlabel = labels[bestfeat]

    featlabels.append(bestfeatlabel)
    myTree = {bestfeatlabel:{}}
    del (labels[bestfeat])

    featvalues = [example[bestfeat] for example in npdata[1:]]
    uniqueVals = set(featvalues)
    for v in uniqueVals:
        myTree[bestfeatlabel][v] = CreateTree(SplitDataSet(npdata,bestfeat,v),labels,featlabels)
    return myTree
def Classify(inputtree,featlabels,testVect):
    firststr = next(iter(inputtree))
    secondDict = inputtree[firststr]
    featIndex = featlabels.index(firststr)
    classLabel = None
    for key in secondDict.keys():
        if testVect[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = Classify(secondDict[key], featlabels, testVect)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__=='__main__':
    '''
    下面注释掉的是直接调的库
    '''
    # Data, Tags = Load_data("mushroom.csv")
    # length = len(Data)
    # trainnum  = int(length*0.8)
    # testnum  = length-trainnum
    # x_one_shot_train = Data[:trainnum]
    # y_Tags = Tags[:trainnum]
    # x_one_shot_test = Data[trainnum:]
    # y_test_Tage = Tags[trainnum:]
    # clf = DecisionTreeClassifier()
    # clf.fit(x_one_shot_train,y_Tags)
    # print(np.mean(clf.predict(x_one_shot_test)==y_test_Tage))
    data = pd.read_csv(path +"mushroom.csv")
    length = len(data)
    trainnum = int(length * 0.8)
    testnum = length - trainnum
    traindata = data[:trainnum]
    testdata = data[trainnum:]
    tags = data.columns
    featLabels = []
    mytree =  CreateTree(data,tags,featLabels)
    print(mytree)
    test = ['s','n']#,'c','n','k','e','e','s','s','w','w','p','w','o','p','k','s','u']
    res = Classify(mytree,featLabels,test)
    print(res)
    del data["class"]

