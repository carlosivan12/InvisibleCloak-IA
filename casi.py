import cv2
from ultralytics import YOLO
import numpy as np

# Iniciar la cámara (0 representa la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Capturar la imagen de fondo sin el objeto que deseas hacer invisible
background = cv2.imread('221013_Haewon_(NMIXX)_Airport_Departure.jpg')  # Reemplaza con tu imagen de fondo
background = cv2.resize(background, (640, 480))  # Ajusta el tamaño según tu cámara

# Leer nuestro modelo YOLO
model = YOLO("best.pt")

while True:
    # Capturar un cuadro de la cámara
    ret, frame = cap.read()

    if not ret:
        break

    # Realizar la detección y segmentación
    resultados = model.predict(frame, imgsz=640, conf=0.78 )

    # Mostrar resultados
    for result in resultados:
        masks = result.masks  # Obtener las máscaras de segmentación

    
    # Convertir las máscaras en una matriz NumPy
    import torch

    # Supongamos que masks es de tipo torch.Size([1, 480, 640])
    #masks = torch.Size([1, 480, 640])

    masks_tensor = torch.tensor(masks.xy)

     
    # Convierte todos los valores distintos de cero a 1
    binary_mask = torch.where(masks_tensor > 0, torch.tensor(1), torch.tensor(0))

    # Convierte el tensor resultante en un array NumPy
    binary_mask_np = cv2.resize(binary_mask.squeeze(0).byte().numpy(), (640, 480))



    # Verifica la forma resultante
    print("Forma del array NumPy:", binary_mask_np.shape)
    print("Array NumPy:\n", binary_mask_np)


    # Tomar la parte de la imagen de fondo que coincide con la máscara de segmentación
    background_masked = cv2.bitwise_and(background, background, mask=binary_mask_np)

    # Combinar ambas partes para obtener el efecto deseado
    final_result = cv2.add(frame, background_masked)

    # Mostrar el resultado
    cv2.imshow('Invisibility Cloak', final_result)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()