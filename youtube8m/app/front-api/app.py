from flask import Flask, render_template, request, make_response, jsonify
from google.cloud import storage
import json
import uuid
import requests
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import subprocess
import csv
import os

VIDEO_LOG_FILE = '/tmp/video.json'
BUCKET_NAME = os.environ["BUCKET_NAME"]
JOBS_PATH = 'data/youtube-8m/jobs'
FEATURE_PREDICT_URL = 'http://{}/predict'.format(os.environ["FEATURE_SERVER_URL"])
PROJECT = os.environ["PROJECT"]
MODEL_NAME = os.environ.get("MODEL_NAME", 'youtube8mkojo')
MODEL_VERSION = os.environ.get("MODEL_VERSION", 'v1')

CACHE_VOCABULARY = '/tmp/vocabulary.csv'
VOCABULARY_PATH = 'data/youtube-8m/vocabulary.csv'
CACHE_VIDEO_INFO = '/tmp/video_info.json'
VIDEO_INFO_PATH = 'data/youtube-8m/video_info.json'

app = Flask(__name__)

if not os.path.exists(CACHE_VOCABULARY):
    subprocess.check_call(args=["gsutil", "cp", "gs://{}/{}".format(BUCKET_NAME, VOCABULARY_PATH), CACHE_VOCABULARY])
labels = {}
with open(CACHE_VOCABULARY, 'r') as c:
    reader = csv.reader(c)
    next(reader, None)
    for r in reader:
        labels[r[0]] = r[3]

def sync_video_info():
    subprocess.check_call(args=["gsutil", "cp", "gs://{}/{}".format(BUCKET_NAME, VIDEO_INFO_PATH), CACHE_VIDEO_INFO])

def set_result(filename, job_id, public_url, predictions):
    with open(CACHE_VIDEO_INFO, "r") as f:
        video_info = json.loads(f.read())
    element = {"filename": filename, "job_id": str(job_id), "public_url": public_url, "predictions": predictions}
    video_info["list"].insert(0, element)
    json_data = json.dumps(video_info)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(VIDEO_INFO_PATH)
    blob.upload_from_string(json_data, content_type="application/json")

    sync_video_info()
    return None

def get_prediction(feature):
    credentials = GoogleCredentials.get_application_default()
    api = discovery.build(
        "ml",
        "v1",
        credentials = credentials,
        discoveryServiceUrl = "https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json"
    )
    request_data = {"instances": [feature]}
    parent = "projects/{}/models/{}/versions/{}".format(PROJECT, MODEL_NAME, MODEL_VERSION)
    response = api.projects().predict(body = request_data, name = parent).execute()
    print(response)
  
    prediction = response['predictions'][0]
    results = []
    for i, index in enumerate(prediction["predicted_topk"]):
        results.append({"label_id": index, "label": labels["{}".format(index)], "probability": prediction['probabilities'][index]})
    return results

def get_file_ext(filename):
  _, ext = filename.rsplit('.', 1)
  return ext

def upload_storage(filename, file_stream, content_type, is_public):
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(BUCKET_NAME)
  blob = bucket.blob(filename)
  blob.upload_from_string(file_stream, content_type=content_type)
  if is_public:
     blob.make_public()
     return blob.public_url
  else:
     return

def get_feature(job_id, filename):
  base, ext = filename.rsplit('.', 1)
  url = "{}/{}/{}/{}".format(FEATURE_PREDICT_URL, job_id, base, ext)
  response = requests.get(url)
  return response.json()

@app.route("/", methods=["GET"])
def index():
    with open(CACHE_VIDEO_INFO, "r") as f:
        video_info = json.loads(f.read())
    return render_template('index.html', recent=video_info["list"][0], vlist=video_info["list"])

@app.route("/show/<job_id>", methods=["GET"])
def show(job_id):
    with open(CACHE_VIDEO_INFO, "r") as f:
        video_info = json.loads(f.read())
    for e in video_info["list"]:
        if e["job_id"] == job_id:
            recent = e
            break
    return render_template('index.html', recent=recent, vlist=video_info["list"])

@app.route("/upload", methods=["POST"])
def upload():
    if 'video' not in request.files:
        make_response(jsonify({'result': 'video is required.'}))

    video = request.files["video"]
    ext = get_file_ext(video.filename)
    if ext not in ['mp4']:
        make_response(jsonify({'result': 'allowed video formats are mp4 and mov only.'}))

    if ext == 'mp4':
      content_type = 'video/mp4'
    else:
      content_type = 'video/quicktime'
    print("DEBUG: upload file({})".format(video.filename))

    job_id = uuid.uuid4()
    filename = "{}/{}/{}".format(JOBS_PATH, job_id, video.filename)
    public_url = upload_storage(filename, video.stream.read(), content_type, True)
    print("DEBUG: upload storage({})".format(public_url))

    feature = get_feature(job_id, video.filename)
    print("DEBUG: get feature")

    predictions = get_prediction(feature)
    print("DEBUG: get prediction({})".format(predictions))

    set_result(video.filename, job_id, public_url, predictions)

    return make_response(jsonify({'result': 'success.'}))

if __name__ == "__main__":
    sync_video_info()
    app.run(debug=True, port=8000, host="0.0.0.0", threaded=True)
