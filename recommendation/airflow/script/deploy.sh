#!/usr/bin/env bash

ENVIRONMENT_NAME=YOUR_ENVIRONMENT_NAME
LOCATION=us-central1

gcloud composer environments storage dags import \
  --environment ${ENVIRONMENT_NAME}  --location ${LOCATION} \
  --source airflow/dags/dag.py
