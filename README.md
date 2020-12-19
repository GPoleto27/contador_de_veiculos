# Contador de Veículos
Aplicação em Python utilizando YOLO e OpenCV para detecção e contagem de carros e motos em um vídeo. (WIP)

# Instalação e execução

## Clonando o Repositório
    $ git clone https://github.com/GPoleto27/contador_de_veiculos

## Instalando as dependências, configurações e pesos da rede
    $ sudo chmod +x setup.sh
    $ ./setup.sh

## Execute a aplicação
    $ ./main.py

# Customização da aplicação

## Alterando a fonte do vídeo
Dentro de _main.py_ edite
> Altere essa variável para utilizar outros videos ou câmeras.

    video_source = "video.mp4"
Você pode usar seu próprio arquivo de vídeo ou webcam.

Para arquivo, apenas modifique o nome do arquivo, para usar sua webcam, altere para um inteiro que irá indicar o índice de sua webcam.
> (Normalmente, se há apenas uma câmera, basta utilizar o valor 0).

## Alterando a área de interesse
Dentro de _main.py_ edite
> Altere essas variáveis para definir área de interesse.

    start_x = 650
    start_y = 450

    end_x = 850
    end_y = 650
Definindo o início e fim (em x e y) de sua área de interesse.

## Alterando os modelos pré-treinados do YOLO
Dentro de _main.py_ edite
> Altere essas variáveis para utilizar outros modelos pré-treinados do YOLO

    model_cfg = 'yolov3.cfg'
    model_weights = 'yolov3.weights'
    scale = 320

Este repositório já baixa as configurações e pesos para _320_.

Para mais detalhes e downloads de redes pré-treinadas, consulte [YOLO](https://pjreddie.com/darknet/yolo/).

## Alterando a tolerância das detecções
Dentro de _main.py_ edite

> Altere esse variável para alterar a tolerância de confiabilidade do resultado.

> Define o quão confiável um resultado deve ser para não ser descartado.

    confidence_threshold = .5

> Altere esse variável para alterar a tolerância das caixas limitantes sobrepostas.

> Quanto menor, menos caixas (reduza se encontrar muitas caixas sobrepostas, aumente caso esteja ignorando muitas detecções).

    nms_threshold = 0

## Utilizando GPU

> Altere essa variável para True se desejar usar GPU (Necessária GPU NVIDIA e drivers)

    use_gpu = False


# TODO
- Contar quantos carros e quantas motos passaram pelo local