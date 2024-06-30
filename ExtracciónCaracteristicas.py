import os
import cv2
import numpy as np
import pandas as pd
from skimage.util import img_as_ubyte
from scipy.stats import moment

# Función para calcular la matriz de co-ocurrencia manualmente
def calcular_matriz_coocurrencia(imagen, distancia, angulo, niveles):
    coocurrencia = np.zeros((niveles, niveles), dtype=np.float64)

    filas, columnas = imagen.shape

    if angulo == 0:
        dx, dy = distancia, 0
    elif angulo == np.pi / 4:
        dx, dy = distancia, -distancia
    elif angulo == np.pi / 2:
        dx, dy = 0, -distancia
    elif angulo == 3 * np.pi / 4:
        dx, dy = -distancia, -distancia

    for i in range(filas):
        for j in range(columnas):
            if 0 <= i + dy < filas and 0 <= j + dx < columnas:
                intensidad1 = imagen[i, j]
                intensidad2 = imagen[i + dy, j + dx]
                coocurrencia[intensidad1, intensidad2] += 1

    return coocurrencia

# Función para calcular las características de textura
def calcular_caracteristicas_textura(imagen, distancia, angulo, niveles):
    imagen_ubyte = img_as_ubyte(imagen)
    matriz_coocurrencia = calcular_matriz_coocurrencia(imagen_ubyte, distancia, angulo, niveles)
    matriz_coocurrencia = matriz_coocurrencia / np.sum(matriz_coocurrencia)

    media = np.mean(matriz_coocurrencia)
    varianza = np.var(matriz_coocurrencia)
    desviacion_estandar = np.std(matriz_coocurrencia)
    R_normalizado = np.max(matriz_coocurrencia) / (np.sum(matriz_coocurrencia) + 1e-10)
    tercer_momento = moment(matriz_coocurrencia.flatten(), moment=3)
    uniformidad = np.sum(matriz_coocurrencia ** 2)
    entropia = -np.sum(matriz_coocurrencia * np.log2(matriz_coocurrencia + 1e-10))

    vector_caracteristicas = [desviacion_estandar, uniformidad, entropia]

    return vector_caracteristicas

# Función para calcular las características SIFT
def calcular_caracteristicas_sift(imagen):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imagen, None)
    if descriptors is not None:
        return descriptors.flatten()[:128]
    else:
        return np.zeros(128)

# Función para procesar una carpeta de imágenes y extraer características
def procesar_carpeta_imagenes(input_folder, output_csv, distancia, angulo, niveles):
    vectores_caracteristicas = []
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print(f"Error al cargar la imagen: {image_path}")
            continue

        caracteristicas_textura = calcular_caracteristicas_textura(imagen, distancia, angulo, niveles)
        caracteristicas_sift = calcular_caracteristicas_sift(imagen)
        vector_caracteristicas = caracteristicas_textura + caracteristicas_sift.tolist()
        vector_caracteristicas.append(image_file)
        vectores_caracteristicas.append(vector_caracteristicas)

    columnas = ['DesviacionEstandar', 'Uniformidad', 'Entropia'] + [f'SIFT_{i}' for i in range(128)] + ['NombreImagen']
    df_caracteristicas = pd.DataFrame(vectores_caracteristicas, columns=columnas)
    df_caracteristicas.to_csv(output_csv, index=False)

# Configuraciones y ejecución
input_folder = "/content/drive/MyDrive/Segmentacion_final"  # Reemplaza con la ruta a tu carpeta de imágenes
output_csv = "caracteristicas_BG1.csv"  # Reemplaza con la ruta de salida deseada
distancia = 1
angulo = np.pi / 4 # Cambia según sea necesario (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
niveles = 256

procesar_carpeta_imagenes(input_folder, output_csv, distancia, angulo, niveles)
