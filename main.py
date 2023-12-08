import os 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


#Create the model. Im just testing right now with a Sequintial model. Plan on going deeper with the functional api in the future.

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Test and train the model on the given data. Might need to add more images for training purposes

train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_data.flow_from_directory(
    './data',
    target_size =(64,64),
    batch_size = 32,
    class_mode = 'binary',
)

model = create_model()
model.fit(train_generator, epochs=30)
model.save('sativa_indica_model.h5')

#--TEST---#

test_image_path = '/Users/justinb/cnn/image_classifer/data/sativa/images3.jpg'

test_image = image.load_img(test_image_path, target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.0 

#--make a perdiction

result = model.predict(test_image)
if result[0][0] > 0.5:
    prediction = 'Sativa'
else:
    prediction = 'Indica'

print(f'The predicted Cannabis Strain is: {prediction}')