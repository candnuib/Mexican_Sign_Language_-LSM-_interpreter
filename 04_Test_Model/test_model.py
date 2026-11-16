import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================
# RUTAS DE LOS MODELOS Y ETIQUETAS / MODELS AND LABELS PATHS
# =====================================================
# Ruta al modelo de MediaPipe para detección de landmarks de la mano
# Path to MediaPipe hand landmarker detection model
MODELO_MP = "../modelos/hand_landmarker.task"

# Ruta al modelo de Deep Learning entrenado (Keras)
# Path to trained Deep Learning model (Keras)
MODELO_IA = "../modelos/static_letters.keras"

# Ruta al archivo de texto con las etiquetas de las clases (letras)
# Path to text file with class labels (letters)
LABELS = "../modelos/labels.txt"

# =====================================================
# CARGAR LABELS / LOAD LABELS
# =====================================================
# Leer el archivo de texto y almacenar cada etiqueta en una lista
# Read the text file and store each label in a list
with open(LABELS, "r") as f:
    etiquetas = [l.strip() for l in f.readlines()]

# =====================================================
# CARGAR RED NEURONAL / LOAD NEURAL NETWORK
# =====================================================
# Cargar el modelo entrenado previamente en TensorFlow/Keras
# Load the previously trained TensorFlow/Keras model
modelo = tf.keras.models.load_model(MODELO_IA)

# =====================================================
# CONFIGURACIÓN DE MEDIAPIPE / MEDIAPIPE CONFIGURATION
# =====================================================
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Configurar las opciones para la detección de la mano en tiempo real (modo imagen)
# Configure options for real-time hand detection (image mode)
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODELO_MP
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)

# =====================================================
# CONFIGURACIÓN DE LA CÁMARA / CAMERA CONFIGURATION
# =====================================================
# Inicializar la captura de video desde la cámara web predeterminada
# Initialize video capture from the default webcam
cap = cv2.VideoCapture(0)

print("Presiona [ESC] para salir. / Press [ESC] to exit.")

while cap.isOpened():
    ok, frame = cap.read()

    if not ok:
        print("Error al leer el fotograma de la cámara. / Error reading camera frame.")
        break

    # Voltear horizontalmente la imagen para un efecto de espejo natural
    # Horizontally flip the image for a natural mirror effect
    frame = cv2.flip(frame, 1)

    # Convertir el fotograma de BGR (OpenCV) a RGB (MediaPipe)
    # Convert frame from BGR (OpenCV) to RGB (MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detectar los puntos clave de la mano utilizando MediaPipe
    # Detect hand keypoints using MediaPipe
    resultado = detector.detect(mp_image)

    # Verificar si se ha detectado al menos una mano en el fotograma
    # Check if at least one hand is detected in the frame
    if len(resultado.hand_landmarks) > 0:
        mano = resultado.hand_landmarks[0]
        datos = []

        # Extraer las coordenadas X, Y, Z de cada uno de los 21 puntos clave (landmarks)
        # Extract X, Y, Z coordinates of each of the 21 keypoints (landmarks)
        for punto in mano:
            datos.append(punto.x)
            datos.append(punto.y)
            datos.append(punto.z)

        # Construir el vector de entrada con las 63 características para la red neuronal
        # Build the input vector with the 63 features for the neural network
        entrada = np.array(
            [datos],
            dtype=np.float32
        )

        # Realizar la inferencia/predicción con el modelo entrenado
        # Perform inference/prediction with the trained model
        pred = modelo.predict(
            entrada,
            verbose=0
        )

        # Encontrar la clase con mayor probabilidad y calcular su porcentaje de confianza
        # Find the class with the highest probability and compute its confidence percentage
        indice = np.argmax(pred)
        confianza = pred[0][indice] * 100
        letra = etiquetas[indice]

        # -----------------------------------------
        # RENDERIZAR RESULTADOS EN PANTALLA / RENDER RESULTS ON SCREEN
        # -----------------------------------------
        texto = f"{letra} ({confianza:.1f}%)"

        # Mostrar la letra predicha y su confianza en la esquina superior izquierda
        # Display the predicted letter and its confidence in the top-left corner
        cv2.putText(
            frame,
            texto,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        # Dibujar los puntos clave (landmarks) sobre la mano detectada
        # Draw keypoints (landmarks) over the detected hand
        alto, ancho, _ = frame.shape
        for p in mano:
            x = int(p.x * ancho)
            y = int(p.y * alto)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 255),
                -1
            )

    else:
        # Mostrar aviso en pantalla si ninguna mano es detectada
        # Display warning on screen if no hand is detected
        cv2.putText(
            frame,
            "Sin mano / No hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Mostrar la imagen procesada en una ventana interactiva de OpenCV
    # Display the processed image in an interactive OpenCV window
    cv2.imshow(
        "Traductor LSM / LSM Translator",
        frame
    )

    tecla = cv2.waitKey(1) & 0xFF

    # Salir del bucle si se presiona la tecla ESC (Código ASCII 27)
    # Break the loop if ESC key is pressed (ASCII code 27)
    if tecla == 27:
        break

# Liberar los recursos de la cámara web y cerrar todas las ventanas de OpenCV
# Release webcam resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
