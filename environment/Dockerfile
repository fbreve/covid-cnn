# hash:sha256:a38fe63f27534aea1f843ab2c1ee0daa14fbd4ca6b04d476916f00700f237703
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        anaconda \
        numpy \
        pandas \
        pillow \
        scikit-learn \
        tensorflow-gpu \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons \
    tensorflow-gpu
