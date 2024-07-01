import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Rutas de las carpetas
ruta_segmentadas = "Segmentaciones"
ruta_recortadas = "Segmentacion_final"

# Crear la carpeta para guardar las imágenes recortadas si no existe
if not os.path.exists(ruta_recortadas):
    os.makedirs(ruta_recortadas)


# Función para encontrar el objeto más grande en la imagen segmentada y recortarlo
def recortar_objeto_mayor(imagen_np, padding=10):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)

    # Encontrar contornos en la imagen
    contornos, _ = cv2.findContours(imagen_gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        return imagen_np  # Si no hay contornos, regresar la imagen original

    # Encontrar el contorno más grande
    contorno_mayor = max(contornos, key=cv2.contourArea)

    # Obtener el bounding box del contorno más grande
    x, y, w, h = cv2.boundingRect(contorno_mayor)

    # Añadir padding para obtener una máscara cuadrada
    x_inicio = max(x - padding, 0)
    y_inicio = max(y - padding, 0)
    lado = max(w, h) + 2 * padding
    lado = min(lado, imagen_np.shape[0] - y_inicio, imagen_np.shape[1] - x_inicio)

    # Recortar la imagen para obtener solo el objeto mayor
    imagen_recortada = imagen_np[y_inicio:y_inicio + lado, x_inicio:x_inicio + lado]

    return imagen_recortada


# Procesar las imágenes segmentadas
for clase in os.listdir(ruta_segmentadas):
    carpeta_clase_segmentadas = os.path.join(ruta_segmentadas, clase)
    carpeta_clase_recortadas = os.path.join(ruta_recortadas, clase)

    # Crear la carpeta de clase en la carpeta de recortadas si no existe
    if not os.path.exists(carpeta_clase_recortadas):
        os.makedirs(carpeta_clase_recortadas)

    for imagen_nombre in os.listdir(carpeta_clase_segmentadas):
        if imagen_nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
            ruta_imagen_segmentada = os.path.join(carpeta_clase_segmentadas, imagen_nombre)

            # Cargar la imagen segmentada
            imagen_segmentada = cv2.imread(ruta_imagen_segmentada)

            if imagen_segmentada is not None:
                # Recortar la imagen para obtener el objeto mayor
                imagen_recortada = recortar_objeto_mayor(imagen_segmentada, padding=10)

                # Guardar la imagen recortada en color original (RGB)
                ruta_imagen_recortada = os.path.join(carpeta_clase_recortadas, imagen_nombre)
                cv2.imwrite(ruta_imagen_recortada, imagen_recortada)


""""# Verificar la cantidad de imágenes segmentadas guardadas
for clase in clases:
    cantidad_guardadas = len(os.listdir(os.path.join(ruta_guardar, clase)))
    print(f"Clase {clase}: {cantidad_guardadas} imágenes segmentadas guardadas")
    
"""