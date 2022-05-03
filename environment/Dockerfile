# hash:sha256:c5eb33021df40378e15f057fc40b18c9ee755e9de95422156fd165517d75f0d8
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy==1.21.5 \
        pandas==1.4.2 \
        pillow==9.0.1 \
        scikit-learn==1.0.2 \
        tensorflow-addons==0.13.0 \
        tensorflow-gpu==2.5.0 \
    && conda clean -ya
