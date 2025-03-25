# Import the necessary functions
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam



# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize the pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0


# One-hot encode the labels
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)


# Print the shape of the data
# print(f"Training data shape: {x_train.shape}")
# print(f"Test data shape: {x_test.shape}")

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)))  #first layer with 32 filters with 3x3 filter size
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))    
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())      #convert 3d outputs to 1d vector which can be passed to fully connected dense layers

# Add a fully connected layer with 128 neurons
model.add(Dense(128, activation='relu'))

# Add a dropout layer to prevent overfitting
model.add(Dropout(0.5))

# Add the output layer with 100 classes (CIFAR-100)
model.add(Dense(100, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_test, y_test))
