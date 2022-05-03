# hash:sha256:95e08d23fc522fde2203a7272831aaf88f0276f159b6a10b17b7fb8df4454534
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons==0.13.0 \
    tensorflow-gpu==2.5.0
