from keras.models import model_from_json
from keras.preprocessing import image
import os
import numpy as np


#loaded the json files
json_file = open("model.json","r") # mention the json file name 
loaded_model_json = json_file.read()
json_file.close()

#loaded file to configure the model
model = model_from_json(loaded_model_json)
model.load_weights("model file name .h5")
print("loadded model....")

#classfy the testing images
def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name,target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_imag =np.expand_dims(test_imag, axis = 0)
    result = model.predict(test_imag)
    if result[0][0] == 1:
        prediction = "name of the sample 1"
    else:
        prediction = "name of the sample 2"
    print(prediction,img_name)


#get the testing samples
path = "testing samples path and folder name"
files = []
for r, d, f in os.walk(path):
    for file in f:
        if ".jpeg" in files:
            files.append(os.path.join(r,file))

#print the results
for f in files:
    classify(f)
    print("\n")