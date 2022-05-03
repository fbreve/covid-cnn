# hash:sha256:494981244da7a15f0b4fa7badbecc1271f34aa16e5406877e96e3d8a067f28bf
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy==1.21.5 \
        pandas==1.3.5 \
        pillow==7.1.2 \
        scikit-learn==1.0.2 \
        tensorflow \
        tensorflow-addons==0.9.1 \
    && conda clean -ya
