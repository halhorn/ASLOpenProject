FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

COPY . /code

RUN apt-get update && \
    apt-get install --yes python3-pip && \
    pip3 install /code

RUN python3 -m pip install --upgrade --quiet cloudml-hypertune

ENTRYPOINT ["python3", "/code/trainer/task.py"]
