import tensorflow as tf
from flask import Flask, helpers
import os
import wgan_estimator
import numpy as np
from PIL import Image
import io

BATCH_SIZE=36
MODEL_DIR=os.environ.get('MODEL_DIR', "./models")
NOISE_DIMS=os.environ.get("NOISE_DIMS", 64)

def load_model(model_dir):
  params = {
    "model_dir": MODEL_DIR,
    "generator_lr": 0.01, 
    "discriminator_lr": 0.01,
  }
  return wgan_estimator.train_and_evaluate(params, True)

def make_ds():
  return (lambda: tf.data.Dataset.from_tensors(0).repeat().map(lambda _: tf.random.normal([BATCH_SIZE, NOISE_DIMS])))

app = Flask(__name__)
model = load_model(MODEL_DIR)
ds = make_ds()

@app.route("/", methods=["GET"])
def index():
  yield_pred = model.predict(ds)
  predictions = np.array([next(yield_pred) for _ in range(36)])
  print(predictions.shape)
  x = np.uint8(np.concatenate([np.concatenate(predictions[i*6:i*6+6]) for i in range(6)], axis=1)*255)
  image = Image.fromarray(x)
  with io.BytesIO() as output:
    image.save(output, format="png")
    content = output.getvalue()
  response = helpers.make_response(content)
  response.headers['Content-Type'] = 'image/png'
  return response
  #return "hello world"

if __name__ == "__main__":
  app.run(debug=True, port=8000, host="0.0.0.0")
