# hash:sha256:a67145200a6e3a8047aeb0dc6ae869c8199128cf9f7a07e1f62cbaab21082bff
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy \
        pandas \
        pillow \
        scikit-learn \
        tensorflow-addons \
        tensorflow-gpu \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-gpu==2.5.0
