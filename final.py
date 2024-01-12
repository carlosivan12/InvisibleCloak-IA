import cv2
from ultralytics import YOLO


model = YOLO("best.pt")  # Reemplaza "best.pt" con la ubicación de tu modelo

# Iniciar la cámara (0 representa la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Cargar la imagen de fondo
background = cv2.imread('221013_Haewon_(NMIXX)_Airport_Departure.jpg')  # Reemplaza con tu imagen de fondo

while True:
    # Capturar un cuadro de la cámara
    ret, frame = cap.read()

    if not ret:
        break

    # Obtener las predicciones del modelo YOLO en el cuadro de la cámara
    results = model(frame)

    # Obtener las coordenadas de las cajas de detección y las clases
    pred_boxes = results.xyxy[0][:, :4]  # Obtener las coordenadas de las cajas
    pred_classes = results.xyxy[0][:, 4].int()  # Obtener las clases de las predicciones

    for box, class_id in zip(pred_boxes, pred_classes):
        x1, y1, x2, y2 = map(int, box)

        # Dibujar un rectángulo alrededor del objeto en el cuadro de la cámara
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostrar el cuadro con las cajas dibujadas
    cv2.imshow('Object Detection', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
