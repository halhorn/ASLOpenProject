#!/bin/sh
# シングルの設定で回す @ frame
JOBNAME=youtube8m_frame_pad_$1_$(date -u +%y%m%d_%H%M%S)
BUCKET=asl-mixi-project-bucket
OUTDIR=gs://$BUCKET/model/youtube8m/frame/auc_pad_$1/$(date -u +%y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOBNAME \
       --region='us-central1' \
       --package-path=$(pwd)/frame \
       --job-dir=$OUTDIR \
       --staging-bucket=gs://$BUCKET \
       --config=$(pwd)/frame/single.yaml \
       -- \
       --train_data_path=gs://$BUCKET/data/youtube-8m-frame/train/train*.tfrecord \
       --eval_data_path=gs://$BUCKET/data/youtube-8m-frame/validate/*.tfrecord \
       --output_dir=${OUTDIR} \
       --model=$1 \
       --train_steps=100000 \
       --pre_layer_num=4 \
       --layer_num=2 \
       --hidden_dim=490 \
       --dropout=0.006 \
       --kernel_regularizer=0.001 \
       --batch_size=112 \
       --learning_rate=0.00023 \
       --category_weight=1.789
