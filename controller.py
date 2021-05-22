from dataset import dataset
import imgCapturer
from sklearn.svm import SVC

class controller:
    def __init__(self):
        self.ds=dataset()
        self.img=None
    
    def getLocalImg(self,direction):
        self.img=imgCapturer.fromLocal(direction)
        return self.img
    
    def getImgFromCamera(self):
        self.img=imgCapturer.fromCamera()
        return self.img
    
    def addImg(self,label):
        self.ds.addImg(self.img,label)
    
    def predict(self,model="SVM"):
        return self.ds.predict(SVC(kernel='linear', probability=True),self.img)