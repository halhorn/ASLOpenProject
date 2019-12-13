From ubuntu:18.04
MAINTAINER HIDETAKA Kojo <hidetaka.kojo@gmail.com>

RUN apt-get -y update
RUN apt-get install -y git curl gnupg
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get -y update && apt-get install -y bazel

RUN git clone --depth 1 https://github.com/google/mediapipe.git
WORKDIR /mediapipe

RUN apt-get install -y libopencv-core-dev libopencv-highgui-dev \
                    libopencv-calib3d-dev libopencv-features2d-dev \
                    libopencv-imgproc-dev libopencv-video-dev

ENV GLOG_logtostderr=1

RUN apt-get install -y python-pip
RUN pip install absl-py
RUN pip install tensorflow

RUN mkdir -p /tmp/mediapipe
WORKDIR /tmp/mediapipe
RUN curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
RUN curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
RUN curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
RUN curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb
RUN curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
RUN tar -xvf /tmp/mediapipe/inception-2015-12-05.tgz

WORKDIR /mediapipe
RUN python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
# RUN bazel build -c opt  --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true  mediapipe/examples/desktop/youtube8m:extract_yt8m_features
bazel build -c opt \
  --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
  mediapipe/examples/desktop/youtube8m:extract_yt8m_features
