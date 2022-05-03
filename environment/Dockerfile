# hash:sha256:5b9b29036248174724cf4739ebd16125647e34896f8cbb152227d439d64bf2b0
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy \
        pandas \
        pillow \
        scikit-learn \
        tensorflow-addons \
    && conda clean -ya
