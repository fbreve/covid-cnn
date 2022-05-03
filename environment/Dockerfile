# hash:sha256:b2bdae06a2240a758a9a57e92a15477ce86518210833d8b92fad4239d8a45174
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
    cudatoolkit==11.3.1 \
    cudnn==8.2.1 \
    tensorflow-addons==0.13.0 \
    tensorflow-gpu==2.5.0
