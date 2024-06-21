import os
import cv2
import matplotlib.pyplot as plt

def recortar_objeto(image, bbox):
    alto, ancho, _ = image.shape

    # Desnormalizar las coordenadas
    x_centro, y_centro, bbox_width, bbox_height = bbox
    x_centro = int(x_centro * ancho)
    y_centro = int(y_centro * alto)
    bbox_width = int(bbox_width * ancho)
    bbox_height = int(bbox_height * alto)

    # Calcular las coordenadas del cuadro delimitador
    x_min = int(x_centro - bbox_width / 2)
    y_min = int(y_centro - bbox_height / 2)
    x_max = int(x_centro + bbox_width / 2)
    y_max = int(y_centro + bbox_height / 2)

    # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(ancho, x_max)
    y_max = min(alto, y_max)

    # Recortar la imagen
    objeto_recortado = image[y_min:y_max, x_min:x_max]

    return objeto_recortado

def visualizar_objeto(image_path, txt_path):
    # Leer la imagen
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None

    # Leer las coordenadas desde el archivo .txt
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split()
            class_id = int(elements[0])
            bbox = tuple(map(float, elements[1:5]))  # (x_centro, y_centro, ancho, alto)

            # Recortar la imagen
            cropped_image = recortar_objeto(image, bbox)

            # Mostrar la imagen recortada
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Imagen (recortada): {os.path.basename(image_path)}')
            plt.axis('off')
            plt.show()

            # Devolver la imagen recortada (opcional)
            return cropped_image

def visualizar_imagen(data_path, n):
    # Obtener la lista de subcarpetas
    class_names = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]

    # Iterar sobre cada subcarpeta (clase)
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        image_names = [file for file in os.listdir(class_path) if file.lower().endswith('.jpg')]

        print(f"Mostrando {n} imágenes recortadas de la clase: {class_name}")

        # Mostrar n imágenes recortadas
        for i, image_name in enumerate(image_names[:n]):
            image_path = os.path.join(class_path, image_name)
            txt_path = os.path.join(class_path, image_name.replace('.jpg', '.txt'))

            if os.path.exists(txt_path):
                visualizar_objeto(image_path, txt_path)
            else:
                print(f"No se encontró el archivo {txt_path}")

# Uso de la función
data_path = "Dataset"  # Cambia esto por la ruta a tu carpeta de datos
n = 1  # Número de imágenes a mostrar de cada clase
visualizar_imagen(data_path, n)



