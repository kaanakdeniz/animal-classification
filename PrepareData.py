import os
import cv2
import numpy as np
import random
import pickle

#Veri setinin bulunduğu dizin atanıyor.
DATADIR = "D:\\Okul\\Yaz Okulu\\Deep Learning\\Proje\\veri"
IMG_SIZE=64
training_data=[]
CATEGORIES=os.listdir(DATADIR)
 
#Veri setindeki her bir kategorinin içindeki veriler ayrı ayrı okunup,
#boyutlandırılıp, etiketleniyor. Modeli eğitecek diziye ekleniyor.
stage=1
for category in CATEGORIES:
    path= os.path.join(DATADIR,category)
    class_num= CATEGORIES.index(category)    
    
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        resized_ImgArray=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE),cv2.IMREAD_GRAYSCALE)
        training_data.append([resized_ImgArray, class_num])
    
    print("Verinin %",(stage/np.size(CATEGORIES))*100,"kadarı okundu.")
    stage+=1
    
#Veri karıştırılıyor.       
random.shuffle(training_data)

X=[]
y=[]

#Karıştırılan veri özellik ve etiket olmak üzere iki farklı diziye atanıyor.
for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Hazırlanmış veri kullanılmak üzere kaydediliyor.
#X->Features  
#y->Labels
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


