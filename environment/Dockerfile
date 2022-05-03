# hash:sha256:812060d1bc46883081cf0ffa81b90a07e82e6e4d28478b584bb51a9608c04604
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy==1.21.5 \
        pandas==1.3.5 \
        pillow==7.1.2 \
        scikit-learn==1.0.2 \
        tensorflow-addons==0.9.1 \
        tensorflow-gpu==2.1.0 \
    && conda clean -ya
