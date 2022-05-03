# hash:sha256:173a35018b1b2572b14a630716eaa948f1c0b78c5cb3d2df79124dc0df86c4f0
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy==1.21.5 \
        pandas==1.3.4 \
        tensorflow-gpu \
    && conda clean -ya
