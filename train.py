from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#get the folder details and training details
trainsetpath = str(input("seclect your train data set path and folder name : "))
valsetpath = str(input("seclect your validation set path and folder name : "))

#build the model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = "relu"))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics =['accuracy'])

#setup the traing and validation 
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

#set to the training and validation 
training_set = train_datagen.flow_from_directory(trainsetpath,target_size =(64,64),batch_size = 12, class_mode = "binary") 
val_set =test_datagen.flow_from_directory(valsetpath, target_size = (64,64),batch_size= 12,class_mode = "binary")

#after training the neural network to fit the model
model.fit_generator(training_set, steps_per_epoch = 10, epochs = 30,validation_data = val_set,validation_steps = 2)

#finally save the training the data form json file
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("save model to disk")
