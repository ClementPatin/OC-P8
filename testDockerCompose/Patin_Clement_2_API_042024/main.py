# imports
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
import tensorflow as tf
import keras

import os
os.environ["SM_FRAMEWORK"] = 'tf.keras'

import segmentation_models as sm


import json
import numpy as np

# initiate the app
app = FastAPI()




# create index
@app.get('/')
def index() :
    return {"message" : "welcome to the Future Vision Transport API"}

# load model interpreter form TfLite folder
interpreter_loaded = tf.lite.Interpreter(model_path="TfLite/resnet34_prod.tflite")


# create predict
@app.post('/predict')
async def predict_mask(img : UploadFile = File(...)) :
    # handle errors
    extension = img.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension :
        raise HTTPException(status_code=400, detail="file should be an image : 'jpg', 'jpeg' or 'png'")
    
    # read file
    image_bytes = await img.read()
    image = tf.io.decode_image(image_bytes)

    image = keras.preprocessing.image.img_to_array(image)

    image = tf.image.resize(image, size=(256, 2*256))
    
    # preprocess image
    image = sm.get_preprocessing("resnet34")(image)

    # put image in a tensor (for inference compatibility)
    image = tf.expand_dims(image, axis=0)

    # cast to float32 (default tflite dtype)
    image = tf.cast(image, dtype='float32')

    # get interpreter input and output details
    input_details = interpreter_loaded.get_input_details()
    output_details = interpreter_loaded.get_output_details()

    # # resize interpreter input
    # interpreter.resize_tensor_input(
    #     input_index=input_details[0]["index"], 
    #     tensor_size=[len(X),text_vectorizer.output_shape[1]]
    #     )
    # allocate
    interpreter_loaded.allocate_tensors()
    # predict scores
    interpreter_loaded.set_tensor(tensor_index=input_details[0]["index"],value=image)
    interpreter_loaded.invoke()

    predicted_scores = interpreter_loaded.get_tensor(output_details[0]['index'])[0]

    # pick the predicted class and create a channel
    predicted_mask = np.argmax(predicted_scores, axis=-1)

    color_map = np.array([
        [  0,   0,   0],
        [128,  64, 128],
        [ 70,  70,  70],
        [153, 153, 153],
        [107, 142,  35],
        [ 70, 130, 180],
        [220,  20,  60],
        [  0,   0, 142]
        ])
    
    predicted_mask_colored = color_map[predicted_mask]

    return {"mask" : json.dumps(predicted_mask_colored.tolist())}







# if __name__ == '__main__' :
#     uvicorn.run(app, host="127.0.0.1", port = 8000)

# uvicorn main:app --reload