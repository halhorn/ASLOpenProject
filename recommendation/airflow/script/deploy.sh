#!/usr/bin/env bash

ENVIRONMENT_NAME=dev-recommendation
LOCATION=us-central1

gcloud composer environments storage dags import \
  --environment ${ENVIRONMENT_NAME}  --location ${LOCATION} \
  --source airflow/dags/dag.py

gsutil cp -r dataflow/extract.py gs://us-central1-dev-recommendat-1e002bc9-bucket/dags/dataflow/
