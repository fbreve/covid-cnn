# hash:sha256:c32f8296fb42f02e4464a467c58d0c8b2757100d7b73f0547251056625e1ef1d
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy==1.21.5 \
        pandas==1.3.5 \
        pillow==9.0.1 \
        scikit-learn==1.0.2 \
        tensorflow-gpu==2.2.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons==0.13.0 \
    tensorflow-gpu==2.5.0
