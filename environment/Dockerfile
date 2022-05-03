# hash:sha256:d05522c011de8258c217b28113674fb439d6c4fed8d44ccd4aeafd3db4388d49
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy \
        pandas \
    && conda clean -ya
