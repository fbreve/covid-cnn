# hash:sha256:d6097386d690064f85f5ee032a2397d0b8f0cef8000a9914273182def62fe131
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        cudatoolkit \
        cuddn \
        numpy \
        pandas \
        pillow \
        scikit-learn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons \
    tensorflow-gpu
