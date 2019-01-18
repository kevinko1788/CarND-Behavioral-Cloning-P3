import csv
import cv2
import numpy as np
import tensorflow
import pdb

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Activation
from keras.layers import Dropout, ELU, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocess_image(image):
#will handle cropping in conv layer
#only apply blur and cvt color to RGB, since image received with cv2.imread(), BGR2RGB
    preprocess_image = cv2.GaussianBlur(image, (3,3),0)
    preprocess_image = cv2.cvtColor(preprocess_image, cv2.COLOR_BGR2RGB)
    return preprocess_image
    
def flip_image(image):
    flipped_image = cv2.flip(image,1)
    return flipped_image

def generator(samples, batch_size=1024):
    while True:
        shuffle(samples)

        images = []
        angles = []

        batch_samples = samples[0:batch_size]

        for image_path, angle in batch_samples: 
            image = cv2.imread(image_path)
            argumented_image = preprocess_image(image) 
            images.append(argumented_image)
            angles.append(angle)
            # To improve: should handle flip earlier and shuffle to avoid bias
            images.append(flip_image(argumented_image))
            angles.append(angle*-1.0)
        print(len(images))
        yield shuffle(np.array(images),np.array(angles))

# Load lines from CSV
lines = []
dataPath = './data'
with open(dataPath + '/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    next(reader, None)
    for line in reader:
        lines.append(line)

images_path = []
angles = []

for line in lines:
    for i in range(3):
        source_path = line[i]
        filename    = source_path.split('/')[-1]
        current_path= './data/IMG/' + filename
        images_path.append(current_path)
        angle = float(line[3])
        if i == 0:
            angles.append(angle)
        elif i == 1:
            angles.append(angle + 0.20)
        else:
            angles.append(angle - 0.20)

#split train and validation data 0.2
samples = list(zip(images_path,angles))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples:', len(train_samples))
print('Validation samples:', len(validation_samples))

#generate images
batch_size = 1024
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples,batch_size = batch_size)

#create model
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))   

# # trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))

#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Conv2D(24,(5,5), strides=(2,2)))
model.add(Activation('elu'))

#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Conv2D(36,(5,5), strides=(2,2)))
model.add(Activation('elu'))

#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Conv2D(48,(5,5), strides=(2,2)))
model.add(Activation('elu'))

#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64,(3,3)))
model.add(Activation('elu'))

#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64,(3,3)))
model.add(Activation('elu'))

#flatten image from 2D to side by side
model.add(Flatten())

#layer 6- fully connected layer 1
model.add(Dense(100))
model.add(Activation('elu'))

#Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
model.add(Dropout(0.25))

#layer 7- fully connected layer 1
model.add(Dense(50))
model.add(Activation('elu'))


#layer 8- fully connected layer 1
model.add(Dense(10))
model.add(Activation('elu'))

#layer 9- fully connected layer 1
model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification

# Compile and train the model
# model.compile(loss='mse',optimizer='adam')
model.compile(optimizer=Adam(lr=1e-4), loss='mse')


file_name = 'model.h5'
print('checkpointer')
checkpointer = ModelCheckpoint(file_name, monitor='val_loss', verbose = 1, save_best_only = True)
print('fit_generator')

model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size, validation_data=validation_generator,  validation_steps=len(validation_samples)//batch_size, epochs=3, verbose=1, callbacks=[checkpointer])

model.save(file_name)
print('model saved!')

model.summary()
