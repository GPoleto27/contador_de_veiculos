FROM python:3
WORKDIR /usr/src/app
RUN apt-get update
RUN apt-get install -y python3-pip git
RUN git clone https://github.com/GPoleto27/contador_de_veiculos \
    && cd contador_de_veiculos \
    && chmod +x setup.sh \
    && ./setup.sh
