FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-training:1.6.0-gpu-py27-cu101-ubuntu16.04
LABEL author="vadimd@amazon.com"

COPY container_training /opt/ml/code
WORKDIR /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM distr_launcher.py

WORKDIR /
