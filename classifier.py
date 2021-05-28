import numpy as np
from sklearn.svm import SVC

class svm:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)
        self.trained=False

    def train(self,ds):
        if len(np.unique(ds.y))>1:
            self.model.fit(ds.X,ds.y)
            self.trained=True

    def predict(self,tar):
        if self.trained:
            yhat_class = self.model.predict(tar[None])[0]
            yhat_prob = self.model.predict_proba(tar[None])[0]
            return yhat_class,yhat_prob
        else:
            return "others",1.0
    
class dist:
    def __init__(self):
        self.trained=False

    def train(self,ds):
        self.X=ds.X
        self.y=ds.y
        if len(self.y)>0:
            self.trained=True
    
    def cal_dist(self,array0,array1):
        dist = np.sqrt(np.sum(np.square(array0 - array1),axis=-1))
        return dist

    def predict(self,tar):
        if self.trained:
            ds=self.cal_dist(self.X,tar)
            print(ds,self.y)
            pre_dist=np.min(ds)
            if pre_dist>0.8:
                return "others",pre_dist
            else:
                minarg=np.argmin(ds)
                return self.y[minarg],pre_dist
        else:
            return "others",0.0