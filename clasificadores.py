import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  # Importar el clasificador SVM
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Paso 1: Cargar los datos desde el archivo CSV
file_path = 'caracteristicas_combinadas.csv'  # Reemplaza con la ruta correcta
data = pd.read_csv(file_path)

# Paso 2: Preprocesamiento de datos
# Separar características (features) y etiquetas (labels)
X = data.drop('Clase', axis=1)  # Ajusta 'Clase' al nombre de la columna de etiquetas
y = data['Clase']

# Codificar etiquetas si es necesario
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir datos en conjunto de entrenamiento y prueba con estratificación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Diccionario para almacenar los modelos
models = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'MLP': MLPClassifier(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)  # Agregar el clasificador SVM
}

# Paso 3: Entrenamiento y evaluación de modelos
for model_name, model in models.items():
    print(f"Entrenando {model_name}...")
    model.fit(X_train, y_train)

    # Predicción y evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # Cambiar a 'macro' para múltiples clases
    confusion = confusion_matrix(y_test, y_pred)

    # Imprimir resultados
    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print("Matriz de Confusión:")
    print(confusion)
    print("-" * 40)


