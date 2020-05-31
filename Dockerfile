# Base image: https://github.com/aws/sagemaker-mxnet-container/blob/master/docker/1.6.0/py3/Dockerfile.gpu
FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04
LABEL author="vadimd@amazon.com"

RUN pip install gluoncv

########### Sagemaker setup ##########
COPY container_training /opt/ml/code
WORKDIR /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM hvd_launcher.py

########### OpenSHH Config for MPI ##########
RUN mkdir -p /var/run/sshd && \
  sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
  mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys