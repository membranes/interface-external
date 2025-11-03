# Pytorch
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime


# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY /.devcontainer/requirements.txt /app


# Environment
SHELL [ "/bin/bash", "-c" ]


# Setting-up
RUN apt update && apt -q -y upgrade && apt -y install sudo && sudo apt -y install graphviz && \
    sudo apt -y install wget && sudo apt -y install curl && sudo apt -y install unzip && \
    sudo apt install lsof && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp/ && cd /tmp && sudo ./aws/install && cd ~ && \
    pip install --upgrade pip && \
	pip install --requirement /app/requirements.txt --no-cache-dir


# Ports
# GRADIO 7860, Spare 8000
EXPOSE 7860 8000


# Setting-up
ARG GRADIO_SERVER_PORT=7860

ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
ENV GRADIO_SERVER_NAME='0.0.0.0'


# ENTRYPOINT
ENTRYPOINT ["bash"]
