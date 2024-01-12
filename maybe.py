import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo de clasificación de color
model = tf.keras.models.load_model('color_clasifier_model.h5')

# Definir el rango de colores que se detectarán (en este caso, azul celeste)
lower_blue = np.array([90, 50, 50])  # Valor mínimo de azul, verde y rojo
upper_blue = np.array([130, 255, 255])  # Valor máximo de azul, verde y rojo

# Iniciar la cámara (0 representa la cámara predeterminada)
cap = cv2.VideoCapture(0)

while True:
    # Capturar un cuadro de la cámara
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionar la imagen al tamaño esperado por el modelo (150x150)
    frame_resized = cv2.resize(frame, (150, 150))

    # Convertir el cuadro de la cámara redimensionado a espacio de color HSV
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Crear una máscara para el rango de colores azul celeste
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clasificar la máscara usando el modelo
    classified_mask = model.predict(np.expand_dims(mask, axis=0))[0]

    # Aplicar un umbral para determinar qué píxeles son azul celeste
    threshold = 0.5  # Ajusta este umbral según tus necesidades
    blue_celeste_pixels = (classified_mask > threshold).astype(np.uint8) * 255

    # Tomar la parte de la imagen de fondo que coincide con el color azul celeste en la cámara
    background_masked = cv2.bitwise_and(background, background, mask=blue_celeste_pixels)

    # Tomar la parte de la imagen de la cámara que no es azul celeste
    result = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(blue_celeste_pixels))

    # Combinar ambas partes para obtener el efecto deseado
    final_result = cv2.add(result, background_masked)

    # Mostrar el resultado
    cv2.imshow('Invisibility Cloak', final_result)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
