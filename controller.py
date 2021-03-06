from dataset import dataset
from imgCapturer import capturer
from classifier import *

class controller:
    def __init__(self):
        self.ds=dataset()
        self.svc=svm()
        self.dist=dist()
        self.cap=capturer()
    
    def getLocalImg(self,direction):
        img,vec=self.cap.fromLocal(direction)
        return img,vec
    
    def getModel(self,model):
        if model=="SVM":
            return self.svc
        if model=="dist":
            return self.dist

    def getImgFromCamera(self,model):
        csf=self.getModel(model)
        self.img,self.vec=self.cap.fromCamera(csf)
        return self.img,self.vec
    
    def addImg(self,vec,label):
        self.ds.addImg(np.array(vec),label)
    
    def trainModel(self,model):
        csf=self.getModel(model)
        csf.train(self.ds)

    def predict(self,vec,model):
        csf=self.getModel(model)
        labels=[]
        ps=[]
        for v in vec:
            label,p=csf.predict(v)
            labels.append(label)
            ps.append(p)
        if len(vec)==0:
            return ["others"],[0.0]
        return labels,ps
    
    def save(self,direction):
        self.ds.save(direction)
    
    def load(self,direction):
        self.ds.load(direction)
    
    def reset(self):
        self.ds=dataset()