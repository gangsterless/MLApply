import numpy as np
#处理数据的工具类
class DataUtility:
    myset = {"mushroom.csv","datingTestSet2.txt"}
    #查看是不是已有数据集
    @staticmethod
    def is_native(name):
        for native_dataset in DataUtility.myset:
            if native_dataset in name:
                return True
        return False
    @staticmethod
    def get_dataset(name,path,shuffle = True,n_train = None,one_shot = False,**kwargs):
        x = []
        count = 0
        with open (path+name,'r',encoding="utf8") as f:
            if DataUtility.is_native(name):
                for sample in f:
                    x.append(sample.strip().split(","))
            elif name=="bank1.0.txt":
                for sample in f:
                    sample = sample.replace('"',"")
                    sample = sample.strip('\n').split(';')
                    x.append(sample)
                    count+=1
            else:
                return NotImplementedError

        if shuffle:
            np.random.shuffle(x)
        return x
