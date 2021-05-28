import numpy as np
from sklearn.svm import SVC

class svm:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)
        self.trained=False

    def train(self,ds):
        if len(np.unique(ds.y))>1:
            X=np.array(ds.X)
            y=np.array(ds.y)
            self.xMean=np.mean(X,axis=0)
            self.xVar=np.var(X,axis=0)
            X=(X-self.xMean)/self.xVar
            self.model.fit(X,y)
            self.trained=True

    def predict(self,tar):
        if self.trained:
            tar=(tar-self.xMean)/self.xVar
            yhat_class = self.model.predict(tar[None])[0]
            yhat_prob = self.model.predict_proba(tar[None])[0]
            return yhat_class,yhat_prob
        else:
            return "others",1.0
    
class dist:
    def __init__(self):
        self.trained=False

    def train(self,ds):
        X=np.array(ds.X)
        y=np.array(ds.y)
        self.X=X
        self.y=y
        if len(self.y)>0:
            self.trained=True
    
    def cal_dist(self,array0,array1):
        dist = np.sqrt(np.sum(np.square(array0 - array1),axis=-1))
        return dist

    def predict(self,tar):
        if self.trained:
            ds=self.cal_dist(self.X,tar)
            pre_dist=np.min(ds)
            minarg=np.argmin(ds)
            return self.y[minarg],pre_dist
            '''
            if pre_dist>0.8:
                return "others",pre_dist
            else:
                minarg=np.argmin(ds)
                return self.y[minarg],pre_dist
            '''
        else:
            return "others",0.0