# hash:sha256:a3897434067fa37d31ffe3bf25fd27fc8e441e8dd1b0c3f8b668c3e43f742c18
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        cudatoolkit \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    cudnn \
    tensorflow-addons \
    tensorflow-gpu
