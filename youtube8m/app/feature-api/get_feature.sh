#!/bin/bash
  
BUCKET="asl-mixi-project-bucket"
JOB_PATH="data/youtube-8m/jobs"
job_id=$1
video_name=$2
ext=$3
temporary_file="/tmp/${job_id}_${video_name}.${ext}"
temporary_metafile="/tmp/metadata_${job_id}_${video_name}.pb"
output_feature="/tmp/feature_${job_id}_${video_name}.pb"
cd ${HOME}/mediapipe/
gsutil cp "gs://${BUCKET}/${JOB_PATH}/${job_id}/${video_name}.${ext}" ${temporary_file}
sec=`ffmpeg -i ${temporary_file} 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | sed 's@\..*@@g' | awk '{ split($1, A, ":"); sp
lit(A[3], B, "."); print 3600*A[1] + 60*A[2] + B[1] }'`
python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example --path_to_input_video=${temporary_file} --clip_end_tim
e_sec=${sec} --path_to_output_metadata=${temporary_metafile}
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features --calculator_graph_config_file=mediapipe/grap
hs/youtube8m/feature_extraction.pbtxt --input_side_packets=input_sequence_example=${temporary_metafile}  --output_side_packets=output
_sequence_example=${output_feature}
