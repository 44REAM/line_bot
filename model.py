
import tensorflow as tf

IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)

base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = True, weights='imagenet')
return base_model