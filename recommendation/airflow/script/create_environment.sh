#!/usr/bin/env bash

ENVIRONMENT_NAME=dev-recommendation
LOCATION=us-central1

# 変数を読み込む
eval `cat airflow/config/.secrets.conf`
echo ${slack_access_token}

# cloud composerの環境作成
gcloud composer environments create ${ENVIRONMENT_NAME} \
    --location ${LOCATION} \
    --python-version 3

# airflowの環境にライブラリーをインストール
gcloud composer environments update ${ENVIRONMENT_NAME} \
--update-pypi-packages-from-file airflow/config/requirements.txt \
--location ${LOCATION}


gcloud composer environments run \
  --location=${LOCATION} \
  ${ENVIRONMENT_NAME} \
  variables -- \
  --set slack_access_token ${slack_access_token} project_id ${project_id}
