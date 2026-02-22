FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Sistema base
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3-pip \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 3.8 por defecto
RUN ln -sf /usr/bin/python3.8 /usr/bin/python
RUN python -m pip install --upgrade pip

WORKDIR /workspace

# Clonar repo oficial
RUN git clone https://github.com/bdi-lab/InGram.git
WORKDIR /workspace/InGram

# Instalar PyTorch EXACTO del repo (CUDA 11.3)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.12.1+cu113

# Instalar resto de dependencias
RUN pip install numpy==1.23.4 scipy==1.9.3 tqdm==4.64.1 python-igraph==0.10.2

CMD ["/bin/bash"]