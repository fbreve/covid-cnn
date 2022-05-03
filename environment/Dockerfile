# hash:sha256:05ca407af8e6fff1764082cdd9887eba85385f16a0244ea2ce1851fc6c73ba14
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        anaconda \
        cudatoolkit \
        cudnn \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons \
    tensorflow-gpu
