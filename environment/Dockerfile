# hash:sha256:1f0b7797910a9feafce38b5d514833497a9cc1cf5fe2e8002e9fdafd58d4ec5b
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy \
    && conda clean -ya
