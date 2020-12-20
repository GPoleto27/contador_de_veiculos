#!/usr/bin/env python3

from cv2 import cv2
import numpy as np

# Altere essa variável para utilizar outros videos ou câmeras
video_source = "video.mp4"

# Altere essas variáveis para definir área de interesse
start_x = 650
start_y = 450

end_x = 850
end_y = 650

# Altere essas variáveis para utilizar outros modelos pré-treinados do YOLO
model_cfg = 'yolov3.cfg'
model_weights = 'yolov3.weights'
scale = 320

# Altere esse variável para alterar a tolerância de confiabilidade do resultado
# Define o quão confiável um resultado deve ser para não ser descartado
confidence_threshold = .5

# Altere esse variável para alterar a tolerância das caixas limitantes sobrepostas
# Quanto menor, menos caixas (reduza se encontrar muitas caixas sobrepostas, aumente caso esteja ignorando muitas detecções)
nms_threshold = 0

global cars_counter
cars_counter = 0
global bikes_counter
bikes_counter = 0

global already_counted
already_counted = False

cap = cv2.VideoCapture(video_source)

classes_file = 'coco.names'
class_names = []
with open(classes_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Função para encontrar objetos na imagem
def find_objects(outputs, img):
    # Lê o formato da imagem
    height, width = img.shape[0], img.shape[1]

    # Define listas de caixa limitante, identificadores de classes e valores de confiabilidade de cada objeto detectado
    bounding_boxes = []
    class_ids = []
    confidence_values = []

    # Para cada uma das saídas
    for output in outputs:
        # Para cada detecção da saída
        for detection in output:
            # Retira os 5 primeiros valores (que indicam as propriedades da detecção)
            scores = detection[5:]
            # Verifica a classe mais provável
            class_id = np.argmax(scores)
            # Verifica a confiabilidade do resultado
            confidence = scores[class_id]

            # Se a confiabilidade for maior do que a tolerância, adiciona a detecção na lista, como uma detecção válida
            if confidence > confidence_threshold:
                # Lê a largura e altura
                w, h = int(detection[2] * width), int(detection[3] * height)
                # Lê o centro da detecção (manipulando os dados da caixa limitante)
                x, y = int((detection[0] * width) - (w / 2)
                           ), int((detection[1] * height) - (h / 2))
                # Adiciona nas listas
                bounding_boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    # Aplica um filtro para caixas limitantes sobrepostas, mantendo apenas a mais confiável
    indices = cv2.dnn.NMSBoxes(
        bounding_boxes, confidence_values, confidence_threshold, nms_threshold)

    if len(indices) == 0:
        globals()['already_counted'] = False

    for i in indices:
        i = i[0]
        box = bounding_boxes[i]

        x, y, w, h = (box[i] for i in range(4))

        if not globals()['already_counted']:
            if class_names[class_ids[i]] == 'car':
                globals()['cars_counter'] += 1
                globals()['already_counted'] = True
            elif class_names[class_ids[i]] == 'motorbike':
                globals()['bikes_counter'] += 1
                globals()['already_counted'] = True

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


layer_names = net.getLayerNames()
out_indices = net.getUnconnectedOutLayers()

output_names = [layer_names[i[0] - 1] for i in out_indices]

while True:
    # Lê a imagem
    success, img = cap.read()

    if success:
        # Recorta a área de interesse
        cropped = img[start_y:end_y, start_x:end_x]

        # Desenha um retângulo na área de interesse
        cv2.rectangle(img, (start_x, start_y),
                      (end_x, end_y), (255, 255, 255), 2)

        cv2.putText(img, "Carros: ", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, str(cars_counter), (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(img, "Motos:", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, str(bikes_counter), (150, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Cria um Blob a partir da imagem
        blob = cv2.dnn.blobFromImage(
            cropped, 1/255, (scale, scale),
            [0, 0, 0], 1,
            crop=False
        )

        # Define o Blob como a entrada da Rede
        net.setInput(blob)

        # Lê as camadas de saída
        outputs = net.forward(output_names)

        # Encontra os objetos na imagem
        find_objects(outputs, cropped)

        try:
            # Mostra a imagem computada
            cv2.imshow('Contador', img)
        except:
            break

        # Delay para o próximo frame
        cv2.waitKey(1)

print(cars_counter, " ", bikes_counter)
