# hash:sha256:93eaad1b6d5df53a596ad1c2d642c8378e3e13174c2326b8491fd1c8afe2b829
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    cudnn==8.2.1 \
    tensorflow-addons==0.13.0 \
    tensorflow-gpu==2.5.0
