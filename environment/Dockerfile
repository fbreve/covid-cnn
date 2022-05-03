# hash:sha256:a8f99fe797f212cac09952737648bf6c8d5785e0e9b1f2625a687c4ae9f01ebb
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    cudatoolkit \
    cudnn \
    tensorflow-addons \
    tensorflow-gpu
