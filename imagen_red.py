import cv2
import numpy as np
import tensorflow as tf

# Cargar un modelo pre-entrenado para la detección de objetos (aquí usaremos MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# Definir una función para verificar si el cuadro contiene azul celeste
def is_blue_celeste(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = tf.keras.applications.mobilenet_v2.preprocess_input(resized_frame)
    predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    predicted_class = np.argmax(predictions)
    return predicted_class == 18  # 18 es la clase para "bluebird" en ImageNet

# Iniciar la cámara (0 representa la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Capturar la imagen de fondo sin el objeto que deseas hacer invisible
background = cv2.imread('221013_Haewon_(NMIXX)_Airport_Departure.jpg')  # Reemplaza con tu imagen de fondo
background = cv2.resize(background, (640, 480))  # Ajusta el tamaño según tu cámara

while True:
    # Capturar un cuadro de la cámara
    ret, frame = cap.read()

    if not ret:
        break

    # Verificar si el cuadro contiene azul celeste
    if is_blue_celeste(frame):
        # Tomar la parte de la imagen de fondo que NO es azul celeste
        background_masked = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

        # Combinar ambas partes para obtener el efecto deseado
        final_result = cv2.add(frame, background_masked)
    else:
        final_result = frame

    # Mostrar el resultado
    cv2.imshow('Invisibility Cloak', final_result)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
