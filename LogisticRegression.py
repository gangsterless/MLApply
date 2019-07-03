#逻辑回归模型，使用的是minist数据集但是，由于是二分类问题所以
#tag本来是0的还是0，超过0的设置为1
import time
import  math
import random
import  pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
path = sys.path[0]+"\data\\"
class LogisticRegression(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000
    def predict_(self,x):
        wx = sum([self.w[j]*x[j]for j in range(len(self.w))])
        return int(wx>0)
    def train(self,features,labels):
        #一个列表，元素个数等于列数（包括标签列）
        self.w = [0.0]*(len(features[0])+1)
        correct_count = 0
        time = 0
        #当小于迭代次数
        while time<self.max_iteration:
            #随机选出一行，最后一个元素赋1
            index = random.randint(0,len(labels)-1)
            x = list(features[index])
            x.append(1.0)
            y = 2*labels[index]-1
            wx = sum([self.w[j]*x[j] for j in range(len(self.w))])
            if wx*y>0:
                correct_count+=1
                if correct_count>self.max_iteration:
                    break
                continue
            for i in range(len(self.w)):
                self.w[i]+=self.learning_step*(y*x[i])
    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels
if __name__=='__main__':
    print('start read data')
    time1 = time.time()
    raw_data = pd.read_csv(path+'train_binary.csv',header=0)
    data = raw_data.values
    imgs = data[0::,1::]
    labels = data[::,0]
    train_features,test_features,train_labels,test_labels = train_test_split(
        imgs,labels,test_size=0.33,random_state=23323
    )
    time_2 = time.time()
    print('read data cost ',time_2-time1)
    print('start training')
    p = LogisticRegression()
    print(train_features.shape)
    print(train_labels.shape)
    p.train(train_features,train_labels)
    time_3 = time.time()
    print('training cost ',time_3-time_2)
    test_predict = p.predict(test_features)
    score = accuracy_score(test_labels,test_predict)
    print('the accuracy is ',score)





