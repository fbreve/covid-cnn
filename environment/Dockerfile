# hash:sha256:817f9b3c22c573cd1b2de81b7df45ceec3612d311d780bb0f6b243db69ea36d2
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy==1.21.5 \
        pandas==1.3.5 \
        pillow==7.1.2 \
        scikit-learn==1.0.2 \
        tensorflow-addons==0.9.1 \
        tensorflow-gpu==2.5.0 \
    && conda clean -ya
