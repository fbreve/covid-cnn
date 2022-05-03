# hash:sha256:bc6653626783c46aff333709a26c950a61bc966fed67ef505296496a187f1cb0
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy==latest \
        pandas \
        pillow \
        scikit-learn \
        tensorflow-addons \
        tensorflow-gpu \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-gpu
