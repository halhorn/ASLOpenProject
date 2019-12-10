#!/bin/sh
JOBNAME=youtube8m_$(date -u +%y%m%d_%H%M%S)
BUCKET=asl-mixi-project-bucket
OUTDIR=gs://$BUCKET/model/youtube8m/video/auc/$(date -u +%y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOBNAME \
       --region=us-central1 \
       --module-name=video.task \
       --package-path=$(pwd)/video \
       --job-dir=$OUTDIR \
       --staging-bucket=gs://$BUCKET \
       --scale-tier=BASIC_GPU \
       --config=$(pwd)/video/hyperparam.yaml \
       --runtime-version=1.14 \
       --python-version=3.5 \
       -- \
       --train_data_path=gs://$BUCKET/data/youtube-8m/train/*.tfrecord \
       --eval_data_path=gs://$BUCKET/data/youtube-8m/valid/*.tfrecord \
       --output_dir=${OUTDIR} \
       --model=dnn \
       --train_steps=30000

