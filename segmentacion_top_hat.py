import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2

# Carpeta de las imágenes de entrenamiento
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

# Función para realzar la nitidez de la imagen
def realce_nitidez(imagen):
    realce = ImageEnhance.Sharpness(imagen)
    imagen_nitidez = realce.enhance(2.0)  # Factor de realce
    return np.array(imagen_nitidez)

# Función para detectar bordes en la imagen
def detectar_bordes(imagen):
    imagen_gris = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(imagen_gris, 50, 150)  # Ajustar umbrales para la detección de bordes
    return bordes

# Función para aplicar histeresis y unir bordes
def unir_bordes(bordes, kernel_size=(5, 5)):
    # Crear un kernel para la unión de bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    bordes_unidos = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
    return bordes_unidos

# Función para aplicar el cierre morfológico en los bordes
def aplicar_cierre(bordes, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
    return bordes_cerrados

# Función para aplicar Top Hat en la imagen
def aplicar_top_hat(imagen, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    top_hat = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
    return top_hat

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

# Procesar imágenes con detección de bordes, aplicar histeresis, cierre, Top Hat, restar imágenes y segmentar
kernel_size_cierre = (7, 7)
kernel_size_top_hat = (9, 9)
kernel_size_cierre2 = (21, 21)

for clase in arreglos_imagenes:
    for idx, imagen in enumerate(arreglos_imagenes[clase]):
        imagen_np = np.array(imagen)

        # Detectar bordes
        bordes = detectar_bordes(imagen)

        # Unir bordes utilizando histeresis
        bordes_unidos = unir_bordes(bordes)

        # Aplicar cierre morfológico
        bordes_cerrados = aplicar_cierre(bordes_unidos, kernel_size_cierre)

        # Aplicar Top Hat a la imagen después del cierre
        top_hat = aplicar_top_hat(bordes_cerrados, kernel_size_top_hat)

        # Restar la imagen Top Hat de la imagen con bordes cerrados
        resta = cv2.subtract(bordes_cerrados, top_hat)

        # Cerrar la imagen nuevamente
        imagen_cerrada = aplicar_cierre(resta, kernel_size_cierre2)

        # Obtener máscara de la mayor área blanca
        _, umbral = cv2.threshold(imagen_cerrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno_mayor = max(contornos, key=cv2.contourArea)
        mascara = np.zeros(imagen_cerrada.shape, dtype=np.uint8)
        cv2.drawContours(mascara, [contorno_mayor], -1, 255, thickness=cv2.FILLED)

        # Segmentar la imagen original con la máscara
        imagen_segmentada = cv2.bitwise_and(imagen_np, imagen_np, mask=mascara)

        # Crear carpeta específica para la clase si no existe
        ruta_clase = os.path.join(ruta_guardar, clase)
        if not os.path.exists(ruta_clase):
            os.makedirs(ruta_clase)

        # Guardar la imagen segmentada
        cv2.imwrite(os.path.join(ruta_clase, f'{idx}_segmentada.png'), cv2.cvtColor(imagen_segmentada, cv2.COLOR_RGB2BGR))

# Verificar la cantidad de imágenes segmentadas guardadas
for clase in clases:
    cantidad_guardadas = len(os.listdir(os.path.join(ruta_guardar, clase)))
    print(f"Clase {clase}: {cantidad_guardadas} imágenes segmentadas guardadas")




