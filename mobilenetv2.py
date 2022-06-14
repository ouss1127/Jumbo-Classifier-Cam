
from keras.applications import mobilenet_v2

model = mobilenet_v2.MobileNetV2()
print(len(model.layers))