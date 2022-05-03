# hash:sha256:26239d0af2de4ca335bddcaa1ed59a2da72c0b338a6107e348f530008b67f5ee
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy==1.19.5 \
        pandas==1.3.5 \
        pillow==9.0.1 \
        scikit-learn==1.0.2 \
        tensorflow-gpu==2.5.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-addons==0.13.0 \
    tensorflow-gpu==2.5.0
