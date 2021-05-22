from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras_facenet import FaceNet
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

class dataset:
    def __init__(self):
        self.X=None
        self.labels=None
        self.y=None
        self.out_encoder = LabelEncoder()
        #self.facenet_model = load_model('facenet_keras.h5')
        #self.detector=MTCNN()
        self.embedder = FaceNet()
        self.required_size=(160, 160)
    
    def img2vec(self,img):
        '''
        results = self.detector.detect_faces(img)#人脸检测
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height#得到所有顶点的位置
        face = img[y1:y2, x1:x2]
        face = Image.fromarray(face)
        face = face.resize(self.required_size)#转换大小
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face-mean)/std#归一化
        face = np.expand_dims(face, axis=0)#三维变四维
        X = self.facenet_model.predict(face)#人脸编码为向量
        '''
        detections = self.embedder.extract(img, threshold=0.5)
        X=detections[0]['embedding']
        return X


    def addImg(self,img,label):
        X=self.img2vec(img)
        if self.X is None:
            self.X=np.array([X])
        else:
            self.X=np.append(self.X,X)
        if self.y is None:
            self.labels=np.array([label])
        else:
            self.labels=np.append(self.labels,label)
        in_encoder = Normalizer()
        self.X = in_encoder.transform(self.X)#归一化
        self.out_encoder.fit(self.labels)
        self.y=self.out_encoder.transform(self.labels)

    
    def predict(self,model,img):
        if len(np.unique(self.y))==1:
            class_index = self.y[0]
            predict_names = self.out_encoder.inverse_transform([class_index])
            return predict_names[0],100.0
        else:
            model.fit(self.X,self.y)
            X=self.img2vec(img)
            yhat_class = model.predict(X)
            yhat_prob = model.predict_proba(X)
            class_index = yhat_class[0]#获取分类（数值型）
            class_probability = yhat_prob[0,class_index] * 100#分类的概率
            predict_names = self.out_encoder.inverse_transform([class_index])#获取分类（标称型）
            return predict_names[0],class_probability