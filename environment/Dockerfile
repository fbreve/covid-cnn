# hash:sha256:bdc71249e5ca82316b9003b82b48127bf43e92e0d4eab4d3619a7bcf33ed07ba
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        anaconda \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        numpy==latest \
        pandas \
        pillow \
        scikit-learn \
        tensorflow-addons \
        tensorflow-gpu \
    && conda clean -ya
