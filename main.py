# import the necessary packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = load_model("Plant_Leaf_Model2.h5")

disease = {
    0 : 'Bercak Daun',
    1 : 'Karat Daun',
    2 : 'Daun Sehat',
    3 : 'Hawar Daun'
}


@app.route("/api/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            try:
                # read the image in PIL format
                img = flask.request.files["image"].read()

                # preprocess the image and prepare it for classification
                imge = image.load_img(io.BytesIO(img), target_size=(299, 299))

                # classify the input image and then initialize the list
                # of predictions to return to the client
                x = image.img_to_array(imge)

                x = np.expand_dims(x, axis=0)

                images = np.vstack([x])

                preds = model.predict(images)

                result = np.argmax(preds[0])

                data["predictions"] = disease[result]

                # data["image"] = str(img)

                # # indicate that the request was a success
                data["success"] = True
            except Exception as err:
                data['error'] = '%s' % (err)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.errorhandler(500)
def internal_server_error(e):
    return flask.jsonify(error=str(e)), 500

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()