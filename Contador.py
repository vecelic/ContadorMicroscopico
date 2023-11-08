# Importações necessárias
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

# Função para carregar o modelo
def load_model(model_name):
    model = tf.saved_model.load(str(model_name))
    model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return model

# Função para carregar os rótulos
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

# Função para detecção em tempo real
def detect_objects(image_np, model, labels):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes']
    image_with_detections = image_np.copy()
    for i in range(num_detections):
        if detections['detection_scores'][i] > 0.5:
            class_id = detections['detection_classes'][i]
            bbox = detections['detection_boxes'][i]
            cv2.rectangle(image_with_detections, (int(bbox[1]*image_np.shape[1]), int(bbox[0]*image_np.shape[0])),
                          (int(bbox[3]*image_np.shape[1]), int(bbox[2]*image_np.shape[0])),
                          (0, 255, 0), 2)
            cv2.putText(image_with_detections, f'{labels[class_id]}: {detections["detection_scores"][i]:.2f}',
                        (int(bbox[1]*image_np.shape[1]), int(bbox[0]*image_np.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image_with_detections

# Carregamento do modelo
MODEL_NAME = 'nome_do_modelo'
model = load_model(MODEL_NAME)

# Carregamento dos rótulos
labels = load_labels('path/to/your/labels_map.pbtxt')

# Carregamento da configuração e restauração do checkpoint
configs = config_util.get_configs_from_pipeline_file('path/to/your/model_config.config')
model_dir = 'path/to/your/model_checkpoint_dir'
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'checkpoint' + '/your_checkpoint_number')).expect_partial()

# Inicialização da captura de vídeo
video_capture = cv2.VideoCapture(0)
object_counts = {}

while True:
    ret, frame = video_capture.read()
    detections = detect_objects(frame, model, labels)

    # Visualização das detecções no quadro
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0],
        detections['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.5)

    # Contagem e visualização das detecções no quadro
    for score, class_id in zip(detections['detection_scores'][0], detections['detection_classes'][0]):
        if score >= 0.5:
            class_name = category_index[class_id]['name']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            # Desenho do nome do objeto e da contagem no quadro
            cv2.putText(frame, f'{class_name}: {object_counts[class_name]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Escrita da contagem dos objetos em um arquivo de texto
with open('object_counts.txt', 'w') as file:
    for object_name, count in object_counts.items():
        file.write(f'{object_name}: {count}\n')

# Libere os recursos
video_capture.release()
cv2.destroyAllWindows()