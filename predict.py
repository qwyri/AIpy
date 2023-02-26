from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
 
modelname = "aiv2_genshin.h5"
# modelname = "genshin_v4.h5"
model = load_model(rf'{modelname}')

imagename = "0337171c7dadf8fc7d214c9a397bcfd5"

image = load_img(f'{imagename}.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)

predclass = f"Predicted Class (0 - Ai , 1 - Not): {label[0][0]} | {np.where(label > 0.5, 1,0)}"
confrate = f"Confidence Rate (0% - Ai Generated, 100% - Original): {round((label[0][0])*100)}%"

f = open("result.txt", "a")
f.write(f"\n\n{modelname}\n{imagename}\n{predclass}\n{confrate}")
f.close()