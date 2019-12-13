from flask import Flask
import subprocess
import tensorflow as tf
from flask import jsonify
tf.enable_eager_execution()
app = Flask(__name__)
def get_feature(file_name):
    example = open(file_name, 'rb').read()
    context_features = {
        "RGB/feature/flxxxxx": tf.VarLenFeature(tf.float32),
        "RGB/feature/timestamp": tf.VarLenFeature(tf.int64),
        "RGB/feature/dimensions": tf.VarLenFeature(tf.int64),
        "RGB/feature/rate": tf.VarLenFeature(tf.float32),
        "AUDIO/feature/timestamp": tf.VarLenFeature(tf.int64),
        "AUDIO/feature/dimensions": tf.VarLenFeature(tf.int64),
        "AUDIO/feature/rate": tf.VarLenFeature(tf.float32),
    }
    sequence_features = {
        "RGB/feature/floats": tf.VarLenFeature(tf.float32),
        "AUDIO/feature/floats": tf.VarLenFeature(tf.float32),
    }
    _, seq = tf.parse_single_sequence_example(example, context_features, sequence_features)
    mean_audio = tf.reduce_mean(tf.sparse.to_dense(seq['AUDIO/feature/floats']), axis=-2)
    mean_rgb = tf.reduce_mean(tf.sparse.to_dense(seq['RGB/feature/floats']), axis=-2)
    return {
        'mean_rgb': mean_rgb.numpy().tolist(),
        'mean_audio': mean_audio.numpy().tolist(),
    }
@app.route("/")
def home():
    return "Hellow World"
@app.route("/predict/<job_id>/<video_name>/<ext>")
def predict(job_id, video_name, ext):
    subprocess.check_call(
            args=[
                "bash",
                "/home/student-00-f8d417950356/mediapipe/get_feature.sh",
                job_id,
                video_name,
                ext,
                ])
    feature = get_feature("/tmp/feature_{}_{}.pb".format(job_id, video_name))
    return jsonify(feature)
if __name__ == '__main__':
    app.run(debug=True, port=8080, host="0.0.0.0")
