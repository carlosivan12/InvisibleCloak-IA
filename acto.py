#Usar enviroment mask 

# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")      
 

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Capturar la imagen de fondo sin el objeto que deseas hacer invisible
background = cv2.imread('221013_Haewon_(NMIXX)_Airport_Departure.jpg')  # Reemplaza con tu imagen de fondo
background = cv2.resize(background, (640, 480))  # Ajusta el tamaño según tu cámara



# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados con una confianza del 70% o superior
    resultados = model.predict(frame, imgsz=640, conf=0.70)

    # Verificamos si hay al menos una detección
    if len(resultados) > 0:
        for result in resultados:
            boxes = result.boxes  # Boxes object for bbox outputs
            mask = result.masks  # Masks object for segmentation masks outputs
            
            if mask is not None:  # Verificar si mask es válido
                # Invertir la máscara para seleccionar el color azul celeste
                mask_inverted = cv2.bitwise_not(mask)

                # Tomar la parte de la imagen de la cámara que no es azul celeste
                result = cv2.bitwise_and(frame, frame, mask=mask_inverted)

                # Tomar la parte de la imagen de fondo que coincide con el color azul celeste en la cámara
                background_masked = cv2.bitwise_and(background, background, mask=mask)

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
