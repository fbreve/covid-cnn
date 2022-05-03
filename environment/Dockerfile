# hash:sha256:5a780700f66fee809918ad43c435209f4a564712dca62077f4ea034461390ec7
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy \
        pandas \
        pillow \
        scikit-learn \
        tensorflow-gpu \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons \
    tensorflow-gpu
