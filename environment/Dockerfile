# hash:sha256:a95f96599ccf3420d6a3280d46b06f1fa9d1a2350e0b1ff8d827d67941042744
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy \
        pandas \
        pillow \
        scikit-learn \
        tensorflow-addons \
        tensorflow-gpu \
    && conda clean -ya
