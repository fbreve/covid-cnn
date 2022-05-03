# hash:sha256:35496330156282c2dbd3ff76201f30a9f4ea4e2eedd890c425d66f99e37d5c29
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=defaults --channel=esri --channel=conda-forge \
        numpy==1.19.5 \
        pandas==1.3.5 \
        pillow==9.0.1 \
        scikit-learn==1.0.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    cudatoolkit==11.3.1 \
    cudnn==8.2.1 \
    tensorflow-addons==0.13.0 \
    tensorflow-gpu==2.5.0
