 
FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git \
    && pip install --upgrade pip \
        pylint \
        autopep8 \
        seaborn \
    && conda install -c conda-forge tensorboard

WORKDIR /work