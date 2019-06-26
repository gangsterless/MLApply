import numpy as np
from sklearn.naive_bayes import MultinomialNB
import sys
from sklearn.preprocessing import OneHotEncoder
path =  sys.path[0]+"\data"
#读文件并切分训练集测试集交叉验证集
def file2matix(filename):
    with open(filename,'r') as file:
        data = np.array([line.strip().split(',') for line in file])
    #跳过第一行
    data = data[1:]
    feature = data[:,4]
    np.delete(data,4,axis=1)
    return data,feature
#类别型特征，字符串转数字
def Str2Letter():#太坑比了竟然有问号
    letters = np.array(list("qwertyuiopasdfghjklzxcvbnm?"))
    data = letters[np.random.randint(0,27,10000)]
    features = set(data)
    feat_dict = {c :i for i,c in enumerate(sorted(features))}
    return feat_dict
def transform(data,feat_dict):
    return [feat_dict[c] for c in data]
if __name__=='__main__':
    data,tag = file2matix(path+'\mushroom.csv')
    feat_dict = Str2Letter()
    print(data)
    #字符转数字
    for i in range(len(data)):
        data[i] = transform(data[i], feat_dict)
    #独热编码
    enc = OneHotEncoder()
    one_shot_data =enc.fit_transform(data)
    #测试集分割
    length = len(data)
    print(length)
    n_train, n_cv = int(0.1 * length), int(0.15 * length)
    idx = np.random.permutation(length)
    train_idx, cv_idx = idx[:n_train], idx[n_train:n_train + n_cv]
    test_idx = idx[n_train + n_cv:]
    x_train, x_cv, x_test = one_shot_data[train_idx], one_shot_data[cv_idx], one_shot_data[test_idx]
    y_train,y_cv,y_test = tag[train_idx],tag[cv_idx],tag[test_idx]
    # print(x_train.shape)
    # print(x_test.shape)
    clf = MultinomialNB()
    # clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    #准确度预测，为什么这么准，感觉有问题啊
    res = clf.predict(x_cv)
    for index,value in enumerate(res):
        if value!=y_cv[index]:
            print(str(index)+"  "+str(value))
    print('准确率为：',np.mean(y_cv==res))
