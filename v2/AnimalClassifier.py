from __future__ import division
import os
import numpy as np
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers


#Sınıf etiketleri alınıyor
def get_class_labels(dir):
   
    classes = os.listdir(dir)
    
    return classes

#Resim yolları alınıyor
def get_class_images(classes, dir):
   
    class_paths = []

    for label in classes:
        
        image_paths = np.array([])
        
        class_path = os.path.join(dir, label)

        images = os.listdir(class_path)

        for image in images:
            
            image_path = os.path.join(class_path, image)

            image_paths = np.append(image_paths, image_path)

        class_paths.append(image_paths)
        
    return class_paths



train_dir = 'D:\\Okul\\Yaz Okulu\\Deep Learning\\Proje\\v2\\dataset\\train'
validate_dir = 'D:\\Okul\\Yaz Okulu\\Deep Learning\\Proje\\v2\\dataset\\validate'


# Train resimlerinin sayısı
n_train = 8000

# Validation resimlerinin sayısıs
n_validation = 2000

# Resim boyutları
image_dim = 200

# Girdi şekli
input_shape = (image_dim, image_dim, 3)

# Öğrenme oranı
learning_rate = 0.001

# Batch boyutu
batch_size = 32

# Epoch sayısı
epochs = 10


train_data_generator = ImageDataGenerator(rescale=1./255,
                                          fill_mode='nearest')
validation_data_generator = ImageDataGenerator(rescale=1./255,
                                               fill_mode='nearest')


train_generator = train_data_generator.flow_from_directory(train_dir,
                                                          target_size=(image_dim, image_dim),
                                                          batch_size=batch_size,
                                                          class_mode='categorical')

validation_generator = validation_data_generator.flow_from_directory(validate_dir,
                                                          target_size=(image_dim, image_dim),
                                                          batch_size=batch_size,
                                                          class_mode='categorical')



classes_dictionary = train_generator.class_indices
class_keys = list(classes_dictionary.keys())
n_classes = len(class_keys)


classes = get_class_labels(validate_dir)
image_paths = get_class_images(classes, validate_dir)


#Modelin tanımlanması
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(n_classes, activation='softmax')
])

# Modelin ayarlamalarının yapılması
model.compile(optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit_generator(train_generator,
                    steps_per_epoch=math.floor(n_train/batch_size),
                    validation_data=validation_generator,
                    validation_steps=n_validation,
                    epochs=epochs)


model.save("AC_CNNv1.model")

model.fit_generator(train_generator,
                    steps_per_epoch=math.floor(n_train/batch_size),
                    validation_data=validation_generator,
                    validation_steps=n_validation,
                    epochs=epochs)

model.save("AC_CNNv2.model")



single_class = class_keys[0]

single_class_image_paths = image_paths[0]



def predict(batch_size, image_paths, model):

    images_arr = []
     
    for image_path in image_paths:
        image_pil = load_img(image_path, interpolation='nearest', target_size=(image_dim, image_dim, 3))

        image_arr = img_to_array(image_pil)

        images_arr.append(image_arr)
 
    images = np.array(images_arr)
    
    predictions = model.predict(images, batch_size=batch_size)

    return predictions

def predictions_accuracy(class_keys, label, predictions):

    
    correct_predictions = 0
    
    n_predictions = len(predictions)
    
    for prediction in predictions:
        most_likely_class = np.argmax(prediction)
        
        prediction_label = class_keys[most_likely_class]
        
        if prediction_label == label:
            
            correct_predictions += 1
            
    average = correct_predictions / n_predictions
    
    return average


single_class_predictions = predict(int(n_validation / n_classes), single_class_image_paths, model)


single_class_accuracy = predictions_accuracy(class_keys, single_class, single_class_predictions)

print("Current accuracy of model for class " + single_class + ": " + str(single_class_accuracy))

def plot_prediction(class_keys, image_paths, predictions):

    for index, image_path in enumerate(image_paths):
        most_likely_class = np.argmax(predictions[index])

        prediction_classes = [str(class_keys[prob_index]) + ": " + str(round(prob*100, 4)) + "%" for prob_index, prob in enumerate(predictions[index])]

        subplot_label = "Prediction: " + str(class_keys[most_likely_class]) + "\nProbabilities: " + ', '.join(prediction_classes)

        fig = plt.figure(figsize=(7, 7), tight_layout=True)
        fig.set_facecolor('white')
        
        image_pil = load_img(image_path, interpolation='nearest', target_size=(200,200))

        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image_pil)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(subplot_label)



predict_image_paths = [image_path[0] for image_path in image_paths]

predictions = predict(10, predict_image_paths, model)

plot_prediction(class_keys, predict_image_paths, predictions)

