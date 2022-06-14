#coding=utf-8
import cv2

import enum

# custom modules 
import camera 
import classifier

# enums
class CameraTypes(enum.Enum):
	webcam = "webcam"
	mv = "mindvision"
class FrameworkTypes(enum.Enum):
	tflite = "tensorflow-lite"
	coral = "coral-tflite"
	dummy = "dummy"
	keras = "keras"

# settings
# TODO make this more flexible by moving to a txt or a JSON.
CAMERA_TYPE = CameraTypes.mv
FRAMEWORK = FrameworkTypes.coral
MODEL_PATH = "models/MobileNetV2.tflite"
LABELS_PATH = "models/tm1_labels.txt"

print('main.py execution in progress')

# main
# This script uses OpenCV
def main():
	try:
		cam = instanciateCamera(CAMERA_TYPE, 0)
		classifier = instanciateClassifier(FRAMEWORK, MODEL_PATH, LABELS_PATH)
		# processFrame = True use to process every frame
		with cam:
			while True: #When to start reading camera
				processFrame = inputEvent() # Defaulted to 'r' key.
				frame = cam.processCapture()
				if processFrame == True:
					results = classifier.classify(frame) 
					print(results)
					processFrame = False # Reset variable
					print("Processed Frame")
				print(frame.shape)
				cv2.imshow("Use Ctrl+C in the terminal to end", frame)
				#print(f'Label: {classifier.labels[results[0].id]}, Score: {results[0].score}') #TODO display Results. Not implemented yet

	except Exception as e: 
		print(e)

def instanciateCamera(type, cameraNumber = 0):
	if type == CameraTypes.webcam:
		cam = camera.WebCam(cameraNumber)
	elif type == CameraTypes.mv:
		cam = camera.MVCam(cameraNumber)
	else:
		print("main.py : Camera type wrong. Defaults to webcam")
		cam = camera.WebCam()
	return cam

def instanciateClassifier(type, modelPath, labelsPath):
	if type == FrameworkTypes.coral:
		cam = classifier.Coral(modelPath)
	elif type == FrameworkTypes.tflite:
		cam = classifier.TFLite(modelPath, labelsPath)
	elif type == FrameworkTypes.dummy:
		cam = classifier.Dummy(modelPath, labelsPath)
	elif type == FrameworkTypes.keras:
		cam = classifier.Keras(modelPath, ['Defect', 'Good'])
	else:
		print("main.py : Classifier type wrong. Defaults to coral")
		cam = classifier.Coral()
	return cam

def inputEvent():
	#TODO generalize function for any input event like GPIO
	if cv2.waitKey(2) == ord('r'):
		print('r pressed')
		return True

if __name__ == '__main__':
    main()

