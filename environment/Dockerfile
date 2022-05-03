# hash:sha256:8772f733ca14604128d90f9976f4d117b3c282a3183dab91e513d90c1429ff40
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
    cudnn==8.2.1 \
    tensorflow-addons==0.13.0 \
    tensorflow-gpu==2.5.0
