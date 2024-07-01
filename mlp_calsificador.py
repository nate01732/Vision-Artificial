import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

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

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en conjunto de entrenamiento y prueba con estratificación
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Definir el clasificador MLP
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Definir el rango de hiperparámetros para el Grid Search
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

# Configurar el Grid Search con validación cruzada
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Entrenar el modelo con Grid Search
print("Realizando Grid Search...")
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros y el mejor score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Mejores parámetros encontrados:", best_params)
print(f"Mejor score de validación (accuracy): {best_score:.5f}")

# Evaluar el modelo con los mejores parámetros en el conjunto de prueba
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
confusion = confusion_matrix(y_test, y_pred)

# Imprimir resultados
print(f"\nEvaluación del mejor modelo en el conjunto de prueba")
print(f"Accuracy: {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print("Matriz de Confusión:")
print(confusion)

