import os
import csv
import cv2
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================================
# CONFIGURACIÓN DE RUTAS / PATH CONFIGURATION
# ==========================================================
# Obtener la ruta base del proyecto de forma dinámica / Get base path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas / Relative paths
DATASET_PATH = os.path.join(BASE_DIR, "../dataset")
MODEL_PATH = os.path.join(BASE_DIR, "../models/hand_landmarker.task")
OUTPUT_PATH = os.path.join(BASE_DIR, "../landmarks")

# Validar que el modelo exista / Validate that the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Error crítico: No se encontró el modelo en '{MODEL_PATH}'. "
        f"Asegúrate de colocar 'hand_landmarker.task' dentro de la carpeta 'models/'."
    )

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================================
# CARGAR MODELO DE MEDIAPIPE / LOAD MEDIAPIPE MODEL
# ==========================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35
)

detector = vision.HandLandmarker.create_from_options(options)

# ==========================================================
# RECORRER DATASET / ITERATE THROUGH DATASET
# ==========================================================
total_imagenes = 0
correctas = 0
fallidas = 0

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"No se encontró la carpeta de dataset en: {DATASET_PATH}")

carpetas = sorted(os.listdir(DATASET_PATH))

for letra in carpetas:
    carpeta_actual = os.path.join(DATASET_PATH, letra)

    if letra.lower() == "none":
        continue
    if not os.path.isdir(carpeta_actual):
        continue

    print(f"\nProcesando letra / Processing class: {letra}")

    salida_csv = os.path.join(OUTPUT_PATH, f"{letra}.csv")

    with open(salida_csv, "w", newline="") as archivo_csv:
        escritor = csv.writer(archivo_csv)
        
        # Crear encabezado con el nombre del archivo, la letra y 21 landmarks (x, y, z)
        # Create header with filename, letter, and 21 landmarks (x, y, z)
        encabezado = ["archivo", "letra"]
        for i in range(21):
            encabezado.append(f"x{i}")
            encabezado.append(f"y{i}")
            encabezado.append(f"z{i}")

        escritor.writerow(encabezado)

        imagenes = sorted(os.listdir(carpeta_actual))

        for nombre in tqdm(imagenes):
            if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total_imagenes += 1
            ruta = os.path.join(carpeta_actual, nombre)
            imagen = cv2.imread(ruta)

            if imagen is None:
                fallidas += 1
                continue

            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)

            resultado = detector.detect(mp_image)

            if not resultado.hand_landmarks:
                print(f"No detectó mano en / Hand not detected in: {nombre}")
                fallidas += 1
                continue

            hand = resultado.hand_landmarks[0]
            fila = [nombre, letra]

            for punto in hand:
                fila.append(punto.x)
                fila.append(punto.y)
                fila.append(punto.z)

            escritor.writerow(fila)
            correctas += 1

# ==========================================================
# RESUMEN / SUMMARY
# ==========================================================
print("\n====================================")
print("PROCESO TERMINADO / PROCESS COMPLETED")
print("====================================")
print(f"Imágenes procesadas / Processed images : {total_imagenes}")
print(f"Correctas           / Successful       : {correctas}")
print(f"Fallidas            / Failed           : {fallidas}")
print(f"\nLos CSV fueron guardados en / CSVs saved in:\n{OUTPUT_PATH}")
