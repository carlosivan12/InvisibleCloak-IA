import cv2
import os
import time

# Directorios para guardar las fotos
train_dir = 'train'
test_dir = 'test'

# Crear los directorios si no existen
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Función para tomar fotos
def take_photos(num_photos, interval, save_dir):
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    for i in range(num_photos):
        # Capturar un cuadro de la cámara
        ret, frame = cap.read()

        # Generar un nombre único para la imagen
        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(save_dir, f'{timestamp}.jpg')

        # Guardar la imagen en el directorio especificado
        cv2.imwrite(filename, frame)

        # Mostrar un mensaje en la consola
        print(f'Foto {i + 1} tomada y guardada en {filename}')

        # Esperar el intervalo especificado
        time.sleep(interval)

    # Liberar la cámara
    cap.release()

# Tomar 40 fotos en la carpeta 'train' con un intervalo de 2 segundos
take_photos(40, 2, train_dir)

# Tomar 10 fotos en la carpeta 'test' con un intervalo de 2 segundos
take_photos(10, 2, test_dir)
