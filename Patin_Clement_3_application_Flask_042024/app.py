from flask import Flask, render_template, url_for, current_app
import requests
import PIL
from PIL import Image as pimg
import numpy as np
import json

app = Flask(__name__)

images_names = [str(i+1) for i in range(10)]  # List of image names

# API_URL = "http://localhost:8000"
API_URL = "https://testapip8.azurewebsites.net"


@app.route("/")
def index():
    return render_template("index.html", images_names=images_names)

@app.route("/show_image/<image_name>")
def show_image(image_name):
    # Validate image name
    if image_name not in images_names:
        return "Invalid image name!"

    return render_template("result.html", images_names=images_names)


@app.route("/predict/<image_name>")
def predict(image_name):
    # check if images are in images
    if image_name not in images_names:
        return "Invalid image name!"
    
    # build url for this image
    image_path = url_for("static", filename="test_images/"+image_name+".png")
    mask_path = url_for("static", filename="test_masks/"+image_name+".png")

    # request API
    headers = {"accept" : "application/json"}
    files=[
        ('img',(image_name+".png", open("static/test_images/"+image_name+".png", 'rb'), 'image/png'))
    ]
    response = requests.post(url = API_URL+"/predict", headers=headers, files=files)

    # extract image array from response
    predicted_mask = json.loads(response.json()["mask"])

    # save in static
    predicted_mask = pimg.fromarray(np.array(predicted_mask, dtype="uint8"))
    predicted_mask.save("static/predicted_mask/predicted_mask.png")

    # build url for this mask
    predicted_mask_path = url_for("static", filename="predicted_mask/predicted_mask.png")

    return render_template("result.html", images_names=images_names, image_name=image_name, image_path=image_path, mask_path=mask_path, predicted_mask_path=predicted_mask_path)


if __name__ == "__main__":
    app.run(debug=True, port=8501)