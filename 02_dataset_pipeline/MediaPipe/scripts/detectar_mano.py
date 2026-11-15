import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- CONFIGURACIÓN DE RUTAS Y MODELO / MODEL AND PATH CONFIGURATION ---
# Ruta a tu modelo hand_landmarker.task / Path to your hand_landmarker.task model
model_path = "../models/hand_landmarker.task"

# --- CONFIGURACIÓN DE MEDIAPIPE / MEDIAPIPE CONFIGURATION ---
base_options = python.BaseOptions(model_asset_path=model_path)

# IMPORTANTE: Usamos HandLandmarkerOptions en lugar de GestureRecognizerOptions
# IMPORTANT: We use HandLandmarkerOptions instead of GestureRecognizerOptions
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1 # Número máximo de manos a detectar / Maximum number of hands to detect
)
detector = vision.HandLandmarker.create_from_options(options)

# Inicializar la captura de video desde la cámara web / Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Conexiones oficiales de los 21 landmarks de la mano / Official connections of the 21 hand landmarks
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Pulgar / Thumb
    (0,5), (5,6), (6,7), (7,8),       # Índice / Index
    (5,9), (9,10), (10,11), (11,12),   # Medio / Middle
    (9,13), (13,14), (14,15), (15,16), # Anular / Ring
    (13,17), (17,18), (18,19), (19,20),# Meñique / Pinky
    (0,17)                            # Base de la palma / Palm base
]

print("=== INICIANDO DETECTOR DE LANDMARKS / STARTING LANDMARK DETECTOR ===")
print("Presiona [ESC] para salir. / Press [ESC] to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara. / Error: Could not read camera frame.")
        break

    # Voltear la imagen horizontalmente (efecto espejo) / Flip the image horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Convertir BGR a RGB para MediaPipe / Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Obtener el timestamp actual en milisegundos / Get current timestamp in milliseconds
    timestamp = int(time.time() * 1000)
    
    # Ejecutar la detección para video / Run detection for video
    result = detector.detect_for_video(mp_image, timestamp)

    # Procesar los landmarks detectados / Process detected landmarks
    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            
            # Dibujar los puntos clave (landmarks) / Draw keypoints (landmarks)
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Dibujar las líneas del esqueleto / Draw skeleton lines
            for connection in HAND_CONNECTIONS:
                start = hand_landmarks[connection[0]]
                end = hand_landmarks[connection[1]]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar la imagen procesada / Display processed image
    cv2.imshow("MediaPipe Hand Landmarker", frame)

    # Salir con la tecla ESC / Exit with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
