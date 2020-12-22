#! /bin/bash

# Instalando dependências
pip3 install -r requirements

# Baixando pesos e configurações da rede
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# Alterando a permissão de execução
<<<<<<< HEAD
chmod +x main.py
=======
chmod +x main.py
>>>>>>> 094cd477d99d9afe08e7d329b80f43f390242ef6
