import numpy as np

class dataset:
    def __init__(self):
        self.X=[]
        self.y=[]

    def addImg(self,X,label):
        self.X.append(X)
        self.y.append(label)
    
    def save(self,direction):
        np.savez(direction,X=np.array(self.X),y=np.array(self.y))

    def load(self,direction):
        dic=np.load(direction)
        self.X=dic["X"].tolist()
        self.y=dic["y"].tolist()