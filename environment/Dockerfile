# hash:sha256:808764faa29f670de01b19931400fb53686917b420278500511fff8a8d150a5c
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        anaconda \
        cudatoolkit \
        cuddn \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons \
    tensorflow-gpu
