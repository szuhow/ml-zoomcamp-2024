import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite


interpreter = tf.lite.Interpreter(model_path="model_2024_hairstyle_v2.tflite")


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

path = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

img = download_image(path)
img = prepare_image(img, (200, 200))
ar = np.array(img)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
to_predict = datagen.flow(np.array([ar]))