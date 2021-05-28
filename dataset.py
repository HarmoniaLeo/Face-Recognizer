import numpy as np

class dataset:
    def __init__(self):
        self.X=np.array([])
        self.y=np.array([])

    def addImg(self,X,label):
        if len(self.X)==0:
            self.X=np.array(X)
        else:
            self.X=np.vstack([self.X,X])
            print(self.X.shape)
        self.y=np.append(self.y,label)
    
    def save(self,direction):
        np.savez(direction,X=self.X,y=self.y)

    def load(self,direction):
        dic=np.load(direction)
        self.X=dic["X"]
        self.y=dic["y"]