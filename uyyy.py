#https://github.com/Classical-machine-learning/invisiblityCloak
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Cargar pesos pre-entrenados (reemplaza con tu propio modelo entrenado)
model.load_weights('blue_celeste_detection_model.h5')

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un cuadro de la cámara
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionar el cuadro al tamaño del modelo
    input_frame = cv2.resize(frame, (640, 480))

    # Realizar la predicción utilizando el modelo
    prediction = model.predict(np.expand_dims(input_frame, axis=0))

    # Clasificar la región actual como "azul celeste" o "no azul celeste"
    label = np.argmax(prediction)

    # Mostrar el resultado en la ventana de OpenCV
    cv2.putText(frame, "Azul Celeste" if label == 1 else "No Azul Celeste", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == 1 else (0, 0, 255), 2)
    cv2.imshow('Detección de Color', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
