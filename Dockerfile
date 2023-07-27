FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="FELT Labs"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y python3 python3-pip \
    && python3 -m pip install --no-cache-dir --upgrade pip

# If set to nothing, will install the latest version
ARG PYTORCH='2.0.1'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu118'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

COPY ./requirements.txt ./requirements.txt

RUN python3 -m pip install -y -r requirements.txt \
    && python3 -m pip uninstall -y tensorflow flax \
    && python3 -m pip install -U "itsdangerous<2.1.0" \
    && rm -rf /var/lib/apt/lists/* 