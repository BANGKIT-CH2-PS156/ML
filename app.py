import os
from flask import Flask, jsonify,request
import requests as req
from werkzeug.utils import secure_filename #for secure name file like "this file.js" to "this-file.js"
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)
#config extention allow
app.config["ALLOWED_EXTENTIONS"] = set(["png","jpg","jpeg"])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

#function to check file we get
def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENTIONS"]

def map_class(prediction):
    if prediction < 0.5:
        return 'Green Bean Coffee is DEFECT'
    else:
        return 'Green Bean Coffee is GOOD'

model = load_model("GBCG_v2.h5", compile=False)

# routing in flask
@app.route("/")
#function in python
def index():
    return jsonify({
        "status":{
            "code": 200,
            "message": "Success fetching the API ML of CoffeeGit App",
        },
        "data": None
    }), 200


@app.route("/predict", methods = ["GET","POST"])
def prediction():
    if request.method == "POST":
        #client send file with key "image"
        data = request.get_json()
        image = data["image"]
        if image:
            response = req.get(image)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((150, 150))
            x = np.asarray(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            
            #predicting the image
            classes = model.predict(images, batch_size=10)
            class_label = map_class(classes[0][0])

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success Predicting"
                },
                "data": {
                    "result": class_label
                }
            }), 200
        else:
            return jsonify({
            "status":{
                "code":400,
                "message": "Bad Request"
            },
            "data": None
        }), 400
    else:
        return jsonify({
            "status":{
                "code":405,
                "message": "method not allow"
            },
            "data": None
        }), 405

#running like void
if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
