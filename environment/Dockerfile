# hash:sha256:5bb55caec810483e0c36ed3a3423d61ae2835b57f4e6c8f17c12f7eca7845121
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=esri \
        numpy==1.21.5 \
        pandas==1.4.2 \
        pillow==9.0.1 \
        python==3.9.12 \
        scikit-learn==1.0.2 \
        tensorflow-addons==0.13.0 \
        tensorflow-gpu==2.5.0 \
    && conda clean -ya
