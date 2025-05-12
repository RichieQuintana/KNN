# K-Vecinos más Cercanos (K-NN)

#Explicación de los cambios:
#Entrenamiento y prueba: El script ahora está mejor documentado para que los estudiantes entiendan cómo dividir los datos en conjuntos de entrenamiento y prueba, y cómo realizar la predicción.
#Escalado de características: Se ha añadido un paso para normalizar los datos con StandardScaler, lo cual es importante para mejorar el rendimiento de los algoritmos basados en distancias, como el K-NN.
#Matriz de confusión: Se genera la matriz de confusión y se calcula la precisión del modelo para evaluar su rendimiento.
#Visualización: La visualización del conjunto de entrenamiento y prueba ahora incluye una visualización del modelo ajustado sobre la malla de características, lo que ayuda a entender cómo el modelo hace las predicciones.

# K-Vecinos más Cercanos (K-NN) con StudentsPerformance.csv

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargando el dataset
dataset = pd.read_csv('StudentsPerformance.csv')  # Lee el archivo CSV

# Preprocesamiento de datos
# Convertimos la variable objetivo a binaria (completed vs none)
dataset['test preparation course'] = dataset['test preparation course'].map({'completed': 1, 'none': 0})

# Seleccionamos características (puntajes) y variable objetivo
X = dataset[['math score', 'reading score', 'writing score']].values
y = dataset['test preparation course'].values

# Dividiendo el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("Tamaño del conjunto de entrenamiento:", len(X_train))
print("Tamaño del conjunto de prueba:", len(X_test))

# Escalado de características (Feature Scaling)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenando el modelo K-NN
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicción de un nuevo resultado (Ejemplo: math=85, reading=90, writing=88)
new_student = sc.transform([[85, 90, 88]])
resultado = classifier.predict(new_student)
print(f"\nPredicción para nuevo estudiante (math:85, reading:90, writing:88): {'Completó' if resultado[0] else 'No completó'} el curso de preparación")

# Predicción sobre el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluación del modelo
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)
print("\nPrecisión del modelo:", accuracy_score(y_test, y_pred))

# Visualización (considerando solo 2 características para poder graficar)
# Usaremos math score y reading score para la visualización
X_vis = X_train[:, :2]  # Tomamos solo las dos primeras características para visualizar
classifier_vis = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_vis.fit(X_vis, y_train)

# Creando la malla para el gráfico
X_set, y_set = X_vis, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                   np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))

plt.contourf(X1, X2, classifier_vis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label='Completado' if j else 'No completado')
    
plt.title('K-NN (Conjunto de Entrenamiento)\nMath Score vs Reading Score')
plt.xlabel('Math Score (escalado)')
plt.ylabel('Reading Score (escalado)')
plt.legend()
plt.show()

# Visualización para el conjunto de prueba
X_vis_test = X_test[:, :2]
X_set, y_set = X_vis_test, y_test

plt.contourf(X1, X2, classifier_vis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label='Completado' if j else 'No completado')
    
plt.title('K-NN (Conjunto de Prueba)\nMath Score vs Reading Score')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.legend()
plt.show()