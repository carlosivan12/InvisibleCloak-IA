#Usar enviroment mask 

# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")      
 

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados
    resultados = model.predict(frame, imgsz = 640, conf = 0.78)

    # Mostramos resultados
    anotaciones = resultados[0].plot()
    for result in resultados:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        

    print(f' masks: {masks}')
    """  import numpy as np  # Si estás utilizando NumPy

    # Si mask es un ndarray de NumPy:
    print("foma de la mascar")
    print(masks.shape)
    import torch

    # Supongamos que masks es de tipo torch.Size([1, 480, 640])
    #masks = torch.Size([1, 480, 640])

    masks_tensor = torch.tensor(masks.xy)
    # Convierte todos los valores distintos de cero a 1
    binary_mask = torch.where(masks_tensor > 0, torch.tensor(1), torch.tensor(0))

    # Convierte el tensor resultante en un array NumPy
    binary_mask_np = binary_mask.squeeze(0).numpy()

    # Verifica la forma resultante
    print("Forma del array NumPy:", binary_mask_np.shape)
    print("Array NumPy:\n", binary_mask_np) """

    # Ahora, normalized_coords contiene las coordenadas (x, y) normalizadas
    #print("coordenadas normalizadas", masks_tensor)
    #print(masks_tensor.shape)


    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)
    
    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

        

cap.release()
cv2.destroyAllWindows()