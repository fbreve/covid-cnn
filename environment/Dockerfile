# hash:sha256:163e20db778f7aa8431b57179677061d6bc43fb1e01c9ded2b6ff1539e9942c1
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy \
        pandas \
        pillow \
        scikit-learn \
        tensorflow \
        tensorflow_addons \
    && conda clean -ya
