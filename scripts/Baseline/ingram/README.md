# Definición de imágen y contenedor para replicar InGRAM:
https://github.com/bdi-lab/InGram/
https://arxiv.org/pdf/2305.19987

CMD build: 
docker build --no-cache -t ingram .
docker build -t ingram .

CMD run:   
docker run --gpus all -it --rm ingram

Quick verification:
# Verificación rápida GPU
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import igraph; print('iGraph OK')"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Verificación de ejecución rápida
python train.py --data_name NL-25 --num_epoch 2

# Verificación de ejecución completa
python train.py --data_name NL-25
python test.py \
  --data_name NL-25 \
  --target_epoch 10000


CMD build: 
docker rmi ingram