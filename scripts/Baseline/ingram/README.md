# Definición de imágen y contenedor para replicar InGRAM:
https://github.com/bdi-lab/InGram/
https://arxiv.org/pdf/2305.19987

CMD build: 
docker build --no-cache -t ingram .
docker build -t ingram .

CMD run:   
docker run -it --rm ingram

CMD build: 
docker rmi ingram