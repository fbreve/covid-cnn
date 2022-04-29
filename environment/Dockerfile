# hash:sha256:a12c5aa6b34405ee0f79253bfe71c692e6db4e6b03b8b3dcf3c851b2b1819468
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya
