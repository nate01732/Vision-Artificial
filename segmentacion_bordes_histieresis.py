import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Carpeta de las imágenes de entrenamiento y de guardar resultados
ruta_carpeta = "Dataset"
ruta_guardar = "ImagenesSegmentadas"

# Crear carpeta para guardar las imágenes segmentadas si no existe
if not os.path.exists(ruta_guardar):
    os.makedirs(ruta_guardar)

# Nombre de las carpetas y clases es el mismo
clases = ["Bhutan Glory", "Chimaera Birdwing", "Kaiser-i-Hind", "Lange's Metalmark", "Palos Verdes Blue"]
arreglos_imagenes = {class_name: [] for class_name in clases}

# Extensiones de imagen válidas
extensiones = ['.jpg', '.jpeg', '.png', 'JPG']

# Función para detectar bordes en la imagen
def detectar_bordes(imagen):
    imagen_gris = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(imagen_gris, 50, 150)  # Umbrales para la detección de bordes
    return bordes

# Función para aplicar histeresis y unir bordes
def unir_bordes(bordes, kernel_size=(11, 11)):
    # Crear un kernel para la unión de bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    bordes_unidos = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
    return bordes_unidos

# Función para cargar y procesar las imágenes
def cargar_imagenes(ruta_carpeta, clases):
    for clase in clases:
        carpeta_clase = os.path.join(ruta_carpeta, clase)
        for imagen_nombre in os.listdir(carpeta_clase):
            if any(imagen_nombre.lower().endswith(ext) for ext in extensiones):
                ruta_imagen = os.path.join(carpeta_clase, imagen_nombre)
                try:
                    imagen = Image.open(ruta_imagen)
                    arreglos_imagenes[clase].append(imagen)
                except Exception as e:
                    print(f"Error al procesar la imagen {ruta_imagen}: {e}")
    return arreglos_imagenes

# Cargar imágenes
arreglos_imagenes = cargar_imagenes(ruta_carpeta, clases)

# Procesar y guardar imágenes con detección de bordes y histeresis
for clase in clases:
    for idx, imagen in enumerate(arreglos_imagenes[clase]):
        imagen_np = np.array(imagen)

        # Detectar bordes
        bordes = detectar_bordes(imagen)

        # Unir bordes utilizando histeresis
        bordes_unidos = unir_bordes(bordes)

        # Encontrar el contorno más grande
        contornos, _ = cv2.findContours(bordes_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contornos) > 0:
            contorno_mayor = max(contornos, key=cv2.contourArea)

            # Crear una máscara a partir del contorno más grande
            mascara = np.zeros(imagen_np.shape[:2], dtype=np.uint8)
            cv2.drawContours(mascara, [contorno_mayor], -1, 255, thickness=cv2.FILLED)

            # Aplicar la máscara a la imagen original para segmentarla
            imagen_segmentada = cv2.bitwise_and(imagen_np, imagen_np, mask=mascara)

            # Crear carpeta específica para la clase si no existe
            ruta_clase = os.path.join(ruta_guardar, clase)
            if not os.path.exists(ruta_clase):
                os.makedirs(ruta_clase)

            # Guardar la imagen segmentada
            nombre_archivo = os.path.join(ruta_clase, f'{clase}_{idx}_segmentada.png')
            cv2.imwrite(nombre_archivo, cv2.cvtColor(imagen_segmentada, cv2.COLOR_RGB2BGR))

# Verificar la cantidad de imágenes segmentadas guardadas
for clase in clases:
    cantidad_guardadas = len(os.listdir(os.path.join(ruta_guardar, clase)))
    print(f"Clase {clase}: {cantidad_guardadas} imágenes segmentadas guardadas")