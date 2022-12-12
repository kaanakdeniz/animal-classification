import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import utils
#Hazırlanmış veri yükleniyor.
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
pickle_in.close()

IMG_SIZE=64

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(10))
model.add(Activation('sigmoid'))

one_hot_labels=utils.to_categorical(y,num_classes=10)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  

model.fit(X, one_hot_labels, batch_size=32, epochs=3, validation_split=0.3)

X.shape
