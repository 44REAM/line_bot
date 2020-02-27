from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request
import werkzeug.datastructures

import tensorflow as tf
import numpy as np

app = Flask(__name__)

def link_handler(url,IMG_SIZE = 160):
    
    image_reader = tf.image.decode_jpeg(
        requests.get(url).content, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    float_caster = float_caster/255.
    float_caster = tf.image.resize(float_caster,[IMG_SIZE,IMG_SIZE], preserve_aspect_ratio=False)
    float_caster = tf.expand_dims(float_caster, 0)
    return float_caster

def get_model(IMG_SIZE = 160):
    IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)

    base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = True, weights='imagenet')
    return base_model

def predicted(url, IMG_SIZE = 160):
    data = link_handler(url,IMG_SIZE)
    model = get_model(IMG_SIZE)
    result = model(data).numpy()

    return str(np.argmax(result))
    

@app.route('/')
def root():
    return 'Hello World!'


@app.route('/hi')
def hi_user():
    IMG_SIZE = 160
    link = request.args['link']
    response = predicted(link, IMG_SIZE)
    return response


def main(requests):

    with app.app_context():
        headers = werkzeug.datastructures.Headers()
        for key, value in requests.headers.items():
            headers.add(key, value)
        with app.test_request_context(method=requests.method, base_url=request.base_url, path=request.path, query_string=request.query_string, headers=headers, data=request.form):
            try:
                rv = app.preprocess_request()
                if rv is None:
                    rv = app.dispatch_request()
            except Exception as e:
                rv = app.handle_user_exception(e)
            response = app.make_response(rv)
            return app.process_response(type(response))

if __name__ == '__main__':
    predicted('https://abdul.in.th/callback/images/Ua604ad880b949603405ba9c3049ebd42/11490376447556.jpg')