# hash:sha256:1a619c9feaf70ce53d1c9f914bc6a09aeb52796bba54bc667d73c08bda4c1fc5
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy==1.21.6 \
        pandas==1.3.5 \
        pillow==9.0.1 \
        scikit-learn==1.0.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons==0.16.1 \
    tensorflow-gpu==2.8.0
