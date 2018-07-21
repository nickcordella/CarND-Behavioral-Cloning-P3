import os
import csv

# Generate list of sampled data points from driving log
samples = []
with open('./combined_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split off 20% of the samples for validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

BATCH_SIZE=128
CORRECTION = 0.3

# Use generator to avoid memory limitations
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
#               Use data from all 3 camera angles
                center = './combined_data/IMG/'+batch_sample[0].split('/')[-1]
                left = './combined_data/IMG/'+batch_sample[1].split('/')[-1]
                right = './combined_data/IMG/'+batch_sample[2].split('/')[-1]                
                center_image = cv2.imread(center)
                left_image = cv2.imread(left)
                right_image = cv2.imread(right)
#               Convert to RGB for compatibility with drive.py
                center_image_converted =  cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image_converted =  cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image_converted =  cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)                
                center_angle = float(batch_sample[3])
#               Adjust left and right camera angles by "CORRECTION" offset.
                left_angle = center_angle + CORRECTION
                right_angle = center_angle - CORRECTION
                images.extend([center_image_converted, left_image_converted, right_image_converted])
                angles.extend([center_angle, left_angle, right_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 160, 320  # image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D

# The neural network ended up being based largely on NVIDIA's architecture from https://arxiv.org/pdf/1604.07316v1.pdf
# Notable differences were the presence of Dropout layers at several points to avoid overfitting
model = Sequential()
# Images were cropped to ignore everything between horizon and hood of car
model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="tanh"))
model.add(Dropout(0.5))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="tanh"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="tanh"))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation="tanh"))
model.add(Conv2D(64, (3, 3), activation="tanh"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

#  Mean squared error minimized, with Adam optimizer used to account for learning rate
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= 
            len(train_samples)/BATCH_SIZE, validation_data=validation_generator, 
            validation_steps=len(validation_samples)/BATCH_SIZE, epochs=7)
model.save('model.h5')
