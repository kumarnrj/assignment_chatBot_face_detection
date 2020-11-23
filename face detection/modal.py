import cv2
from imutils.video import WebcamVideoStream


class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()
    
    def __del__(self):
        self.stream.stop()
    
    def get_frame(self):
        image = self.stream.read()

        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = detector.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
            
    # Release the VideoCapture object
    

        ret,jpeg = cv2.imencode('.jpg',image)
        data =[]
        data.append(jpeg.tobytes())
        return data

VideoCamera()  

