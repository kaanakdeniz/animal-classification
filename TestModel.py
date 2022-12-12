import tensorflow as tf
import cv2

model =tf.keras.models.load_model("CannyEdgeCNN.model")

def prepare(path,IMG_SIZE=64):
    img_array=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    filtered_img_array=cv2.Canny(img_array,100,200)
    resFil_img_array=cv2.resize(filtered_img_array,(IMG_SIZE,IMG_SIZE))
    return resFil_img_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

predictionClean = model.predict([prepare("DATA\\Test\\1.jpg")])
predictionDirty = model.predict([prepare("DATA\\Test\\2.jpg")])

