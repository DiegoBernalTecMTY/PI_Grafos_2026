# Docker: Imágen y contenedor para replicar InGRAM:
https://github.com/bdi-lab/InGram/
https://arxiv.org/pdf/2305.19987

CMD build: 

docker build --no-cache -t ingram .

or

docker build -t ingram .

CMD run:   

docker run --gpus all -it --rm ingram

CMD remove image: 

docker rmi ingram

# Verificación rápida GPU
python -c "import torch; print('Torch:', torch.__version__)"

python -c "import igraph; print('iGraph OK')"

python -c "import torch; print(torch.cuda.is_available())"

python -c "import torch; print(torch.cuda.get_device_name(0))"

# VM o Linux nativo

Instala python en /home por compatibilidad con GPUs rentadas

bash setup_ingram.sh

Antes de ejecutar InGRAM es necesario usar el python correcto:

export PATH=/home/python3.8/bin:$PATH && hash -r

# Verificación de ejecución rápida
python train.py --data_name NL-25 --num_epoch 2

# Verificación de ejecución completa
python train.py --data_name NL-25

python test.py \
  --data_name NL-25 \
  --target_epoch 10000
