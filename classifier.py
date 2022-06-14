import time
import cv2
import numpy as np
#from tensorflow import lite as tflite


#Coral
#import pycoral
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify

#Keras
#from keras.models import load_model

#Dummy
import random
from time import sleep

class Coral:
    def __init__(self, modelPath):
        self.modelPath = modelPath
#        self.labelsPath = labelsPath
        self.interpreter = make_interpreter(self.modelPath)
        self.interpreter.allocate_tensors()

#        self.labels = read_label_file(self.labelsPath)
      
    def __enter__(self):
        self.interpreter.allocate_tensors()
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        return
    
    def classify(self, image):
        size = common.input_size(self.interpreter)
        common.set_input(self.interpreter, cv2.resize(image, size, fx=0, fy=0,
                                                interpolation=cv2.INTER_CUBIC)) #TODO check size
        self.interpreter.invoke()
        return classify.get_classes(self.interpreter) #results

class TFLite:
    def __init__(self, modelPath, labelsPath):
        self.modelPath = modelPath
        self.labelsPath = labelsPath
        self.interpreter = tflite.Interpreter(model_path=self.modelPath)
        self.interpreter.allocate_tensors()
        # self.signature = self.interpreter.get_signature_runner()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

      
    def __enter__(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path=self.modelPath)
        self.interpreter.allocate_tensors()
        print("Class not implemented")
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        return #should stay empty
    
    def classify(self, image):
        start_time = time.time()


        # print(self.input_details[0]['shape'])
        # print(self.input_details[0]['index'])
        resized_image = cv2.resize(image, (224, 224))
        rescaled_image = np.array(resized_image*1./255, dtype=np.float32) 
        self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(rescaled_image, axis=0))

        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(output_data)
        print("--- %s seconds ---" % (time.time() - start_time))
        return

        # size = pycoral.adapters.common.input_size(self.interpreter)
        # pycoral.adapters.common.set_input(self.interpreter, cv2.resize(image, size, fx=0, fy=0,
        #                                         interpolation=cv2.INTER_CUBIC)) #TODO check size
        # self.interpreter.invoke()
        # return pycoral.adapters.classify.get_classes(self.interpreter) #results

class Dummy:
    def __init__(self, modelPath, labelsPath, processingTime = 0.05):
        self.modelPath = modelPath
        self.labelsPath = labelsPath
        self.time = processingTime
        self.labels = read_label_file(self.labelsPath)
      
    def __enter__(self):
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        return #should stay empty
    
    def classify(self, image):
        # self.interpreter.invoke()
        sleep(self.time)
        print(f'Label: {self.labels[random.randint(0, 1)]}, Score: {random.random()}')
        return [random.randint(0, 1), random.random()]

class Keras:
    def __init__(self, model_path, labels):
        self.model_path = model_path
        self.labels = labels
        self.keras_model = load_model(model_path)
    
    def classify(self, image):
        resized_image = cv2.resize(image, (224, 224))
        rescaled_image = resized_image*1./255
        prediction = self.keras_model.predict(np.expand_dims(rescaled_image, axis=0))
        print(prediction)
        return prediction
    
