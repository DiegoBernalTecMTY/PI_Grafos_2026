#!/bin/bash
set -e

echo "[1/9] Instalando dependencias de compilación"
apt-get update
apt-get install -y build-essential wget git \
    libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev

cd /home

echo "[2/9] Descargando Python 3.8.20"
rm -rf Python-3.8.20 Python-3.8.20.tgz || true
wget https://www.python.org/ftp/python/3.8.20/Python-3.8.20.tgz
tar -xf Python-3.8.20.tgz
cd Python-3.8.20

echo "[3/9] Compilando Python en /home/python3.8"
rm -rf /home/python3.8 || true
./configure --prefix=/home/python3.8 --enable-optimizations
make -j$(nproc)
make install

echo "[4/9] Creando alias python -> python3.8"
ln -sf /home/python3.8/bin/python3.8 /home/python3.8/bin/python

echo "[5/9] Forzando PATH hacia /home/python3.8"
export PATH=/home/python3.8/bin:$PATH
hash -r

echo "[6/9] Verificando Python activo"
python --version

echo "[7/9] Actualizando pip e instalando dependencias"
python -m ensurepip
python -m pip install --upgrade pip

python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.12.1+cu113 \
    numpy==1.23.4 \
    scipy==1.9.3 \
    tqdm==4.64.1 \
    python-igraph==0.10.2

echo "[8/9] Clonando repositorio InGram en /home"
cd /home
rm -rf InGram || true
git clone https://github.com/bdi-lab/InGram.git

echo "[9/9] Verificando instalación Torch y CUDA"
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

echo "[DONE] Entorno InGram completamente listo en /home"

#Antes de ejecutar InGRAM es necesario usar el python correcto:
#export PATH=/home/python3.8/bin:$PATH && hash -r