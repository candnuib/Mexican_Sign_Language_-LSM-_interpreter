# =====================================================
# IMPORTACIÓN DE LIBRERÍAS / LIBRARY IMPORTS
# =====================================================
import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================================
# CONFIGURACIÓN DE RUTAS / PATH CONFIGURATION
# =====================================================
LANDMARKS_DIR = "../landmarks"
MODELOS_DIR = "../modelos"

# Crear la carpeta de modelos si no existe / Create models directory if it doesn't exist
os.makedirs(MODELOS_DIR, exist_ok=True)

# =====================================================
# CARGAR Y CONCATENAR DATASET / LOAD AND CONCATENATE DATASET
# =====================================================
print("=" * 50)
print("CARGANDO DATASET / LOADING DATASET")
print("=" * 50)

dfs = []

# Iterar sobre todos los archivos CSV individuales en la carpeta de landmarks
# Iterate over all individual CSV files in the landmarks folder
for archivo in sorted(os.listdir(LANDMARKS_DIR)):
    if archivo.endswith(".csv"):
        ruta = os.path.join(LANDMARKS_DIR, archivo)
        df = pd.read_csv(ruta)
        dfs.append(df)

# Unir todos los DataFrames en un único conjunto de datos global
# Merge all DataFrames into a single global dataset
dataset = pd.concat(dfs, ignore_index=True)

print("Muestras totales / Total samples:", len(dataset))

# =====================================================
# CODIFICAR ETIQUETAS / ENCODE LABELS
# =====================================================
# Convertir las letras en identificadores numéricos (ej. A -> 0, B -> 1) requeridos por la red
# Convert letters into numerical identifiers (e.g., A -> 0, B -> 1) required by the network
encoder = LabelEncoder()
dataset["label_id"] = encoder.fit_transform(dataset["letra"])

# =====================================================
# SEPARAR ENTRADAS Y SALIDAS / SPLIT INPUTS AND OUTPUTS
# =====================================================
# X contiene las características: las 63 coordenadas (21 landmarks * 3 ejes x, y, z)
# X contains the features: the 63 coordinates (21 landmarks * 3 axes x, y, z)
X = dataset.drop(columns=["archivo", "letra", "label_id"])

# y contiene la etiqueta numérica de la clase objetivo
# y contains the numerical label of the target class
y = dataset["label_id"]

# =====================================================
# DIVISIÓN DEL DATASET (TRAIN / VAL / TEST) / DATASET SPLIT
# =====================================================
# Dividir en Entrenamiento (80%) y Temporal (20%) manteniendo la proporción de clases (stratify)
# Split into Training (80%) and Temporal (20%) keeping class proportions (stratify)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Dividir el conjunto temporal en Validación (10%) y Prueba (10%)
# Split the temporal set into Validation (10%) and Test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nEntrenamiento / Training :", len(X_train))
print("Validación    / Validation   :", len(X_val))
print("Prueba        / Test         :", len(X_test))

# =====================================================
# CREAR ARQUITECTURA DE LA RED NEURONAL / CREATE NEURAL NETWORK ARCHITECTURE
# =====================================================
# Red Neuronal Densa (Multilayer Perceptron) para clasificación multiclase
# Dense Neural Network (Multilayer Perceptron) for multi-class classification
modelo = tf.keras.Sequential([
    # Capa de entrada adaptada a los 63 valores de los landmarks
    # Input layer adapted to the 63 landmark values
    tf.keras.layers.Input(shape=(63,)),

    # Capa oculta densa de 128 neuronas con función de activación ReLU
    # Dense hidden layer of 128 neurons with ReLU activation function
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Capa oculta densa de 64 neuronas con función de activación ReLU
    # Dense hidden layer of 64 neurons with ReLU activation function
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    # Capa de salida con tantas neuronas como clases (letras) existan y activación Softmax
    # Output layer with as many neurons as classes (letters) exist and Softmax activation
    tf.keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])

# =====================================================
# COMPILAR EL MODELO / COMPILE THE MODEL
# =====================================================
modelo.compile(
    optimizer="adam",  # Optimizador basado en descenso de gradiente estocástico adaptativo / Adaptive SGD optimizer
    loss="sparse_categorical_crossentropy",  # Función de pérdida para etiquetas enteras multiclase / Loss function for multi-class integer labels
    metrics=["accuracy"]  # Métrica principal de rendimiento / Main performance metric
)

# Imprimir la estructura de capas y parámetros de la red
# Print the layer structure and network parameters
modelo.summary()

# =====================================================
# CONFIGURAR CALLBACKS / CONFIGURE CALLBACKS
# =====================================================
# 1. ModelCheckpoint: Guarda automáticamente el archivo .keras del mejor modelo según la precisión de validación
# 1. ModelCheckpoint: Automatically saves the .keras file of the best model based on validation accuracy
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(
        MODELOS_DIR,
        "static_letters.keras"
    ),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# 2. EarlyStopping: Detiene el entrenamiento si no hay mejora en la validación durante 10 épocas consecutivas
# 2. EarlyStopping: Stops training if there's no improvement in validation for 10 consecutive epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

# =====================================================
# ENTRENAMIENTO DEL MODELO / MODEL TRAINING
# =====================================================
historial = modelo.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=100,
    batch_size=32,
    callbacks=[
        checkpoint,
        early_stop
    ]
)

# =====================================================
# EVALUACIÓN EN CONJUNTO DE PRUEBA / EVALUATION ON TEST SET
# =====================================================
# Evaluar con datos completamente aislados que el modelo jamás vio durante el entrenamiento
# Evaluate using completely isolated data the model never saw during training
loss, accuracy = modelo.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n")
print("=" * 50)
print("RESULTADO FINAL / FINAL RESULT")
print("=" * 50)
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Loss     : {loss:.4f}")

# =====================================================
# GUARDAR DICCIONARIO DE ETIQUETAS / SAVE LABEL DICTIONARY
# =====================================================
# Exportar las clases a un archivo de texto plano para mapearlas correctamente en la inferencia en vivo
# Export classes to a plain text file to map them correctly during live inference
with open(
    os.path.join(MODELOS_DIR, "labels.txt"),
    "w"
) as f:
    for letra in encoder.classes_:
        f.write(letra + "\n")

print("\nModelo guardado correctamente. / Model successfully saved.")
