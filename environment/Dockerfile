# hash:sha256:04659c6bb37f333c2368ef76638115c9e9fe4bad99e9cd1b5480dde1e1719c61
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy \
        pandas \
        pillow \
        scikit-learn \
        tensorflow \
        tensorflow-addons \
    && conda clean -ya
