from PIL import Image
import cv2
import numpy as np

def fromLocal(direction):
    image = Image.open(direction)
    image = image.convert('RGB')
    return np.asarray(image)

def fromCamera():
    capture = cv2.VideoCapture(0) 
    while(1):
        ret, frame = capture.read()
        cv2.imshow("press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows() 
    return np.asarray(frame)