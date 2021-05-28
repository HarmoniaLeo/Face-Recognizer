from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import detect_face
import cv2
from cal_128XVector_user_facenet import cal_128_vector,build_facenet_model,cal_dist_from_csv

class capturer:
    def __init__(self):
        # 调用facenet模型
        self.sess1, self.images_placeholder, self.phase_train_placeholder, self.embeddings = build_facenet_model()
        image_size = 200
        self.minsize = 20
        self.threshold = [0.4, 0.5, 0.5]
        self.factor = 0.709  # scale factor
        print("Creating MTcnn networks and load paramenters..")
        #########################build mtcnn########################
        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, './model/')

    def toVecs(self,img):
        size=img.shape
        bounding_box, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        imgs=[]
        vecs=[]
        for face_position in bounding_box:
            rect = face_position.astype(int)
            if 0<rect[0]-5<size[1] and 0<rect[1]-5<size[0] and 0<rect[2]+5<size[1] and 0<rect[3]+5<size[0]:
                image=img[rect[1]:rect[3],rect[0]:rect[2]]#截取人脸的ROI区域
                array=cal_128_vector(image,self.sess1, self.images_placeholder, self.phase_train_placeholder, self.embeddings)#计算人脸的128向量
                array=array[0]
                imgs.append(image)
                vecs.append(array)
        return imgs,vecs

    def fromLocal(self,direction):
        img=cv2.imread(direction)
        return self.toVecs(img)

    def fromCamera(self,clf):
        capture = cv2.VideoCapture(0) 
        size = (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        while (capture.isOpened()):
            ret, frame = capture.read()
            frame1=frame.copy()
            bounding_box, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

            nb_faces = bounding_box.shape[0]  # 人脸检测的个数
            # 标记人脸
            for face_position in bounding_box:
                rect = face_position.astype(int)
                if 0<rect[0]-5<size[1] and 0<rect[1]-5<size[0] and 0<rect[2]+5<size[1] and 0<rect[3]+5<size[0]:
                    image=frame[rect[1]:rect[3],rect[0]:rect[2]]#截取人脸的ROI区域
                    array=cal_128_vector(image,self.sess1, self.images_placeholder, self.phase_train_placeholder, self.embeddings)#计算人脸的128向量
                    array=array[0]
                    label,dist=clf.predict(array)
                    # 矩形框
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2, 1)
                    cv2.putText(frame, "faces:%d" % (nb_faces), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 4)
                    cv2.putText(frame, '%.2f' % (dist), (rect[0], rect[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 4)
                    cv2.putText(frame, label, (rect[0], rect[1] ), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 4)
            cv2.imshow("press 'q' to capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows() 
        img=np.asarray(frame1)
        return self.toVecs(img)