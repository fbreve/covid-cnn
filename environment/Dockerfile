# hash:sha256:e005bfe1d56beda8e838e9b51c11f6d6bd6c067415bfab6a07d00ce976532374
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy==1.21.5 \
        pandas==1.3.5 \
        pillow==7.1.2 \
        scikit-learn==1.0.2 \
        tensorflow-addons==0.9.1 \
    && conda clean -ya
