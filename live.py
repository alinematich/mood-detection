import cv2
import numpy as np
from PIL import Image
from keras import models

#Load the saved model
model = models.load_model('model-0.5506.h5')
video = cv2.VideoCapture(0)
categories=['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
# categories=['happy', 'Disgust', 'Fear', 'Happiness', 'angry', 'Surprise', 'Neutral']
size = 48
hight = 640
width = 480
crop = 200

while True:
        _, frame = video.read()

        #Convert the captured frame into Gray    
        frame = frame[int(hight/2-crop/2):int(hight/2+crop/2), int(width/2-crop/2):int(width/2+crop/2)]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Resizing into 128x128 because we trained the model with this image size.
        im = Image.fromarray(frame, 'L')
        im = im.resize((size,size))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=3)

        #Calling the predict method on model to predict 'me' on the image
        # prediction = int(model.predict(img_array)[0][0])
        prediction = model.predict(img_array)[0]
        print(categories[np.argmax(prediction)])

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break

video.release()
cv2.destroyAllWindows()