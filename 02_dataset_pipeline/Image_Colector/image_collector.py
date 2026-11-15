import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================
# CONFIGURACIÓN DE RUTAS / PATH CONFIGURATION
# ============================
# Usando rutas relativas para portabilidad en GitHub / Using relative paths for GitHub portability
DATASET_DIR = "../dataset"
MODEL_PATH = "../models/hand_landmarker.task"

# Validar si el archivo del modelo existe antes de iniciar / Validate if the model file exists before starting
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Error crítico: No se encontró el modelo en '{MODEL_PATH}'. "
        f"Asegúrate de colocar 'hand_landmarker.task' dentro de la carpeta 'models/'."
    )

letra_actual = "A"

# Crear la carpeta del dataset si no existe / Create the dataset directory if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)

# ============================
# CONFIGURACIÓN DE MEDIAPIPE / MEDIAPIPE CONFIGURATION
# ============================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35,
)
detector = vision.HandLandmarker.create_from_options(options)

# Conexiones oficiales para renderizar el esqueleto de la mano / Official connections to render hand skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Inicializar la captura de video (Cámara web) / Initialize video capture (Webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: No se pudo abrir la cámara. / Error: Could not open the webcam.")

print("=========================================================")
print(" RECOLECTOR DE DATASET PARA LSM / DATASET COLLECTOR")
print(" ESPACIO: Guardar (solo si hay mano) | A-Z: Cambiar letra")
print(" 0: Seleccionar 'none' | ESC: Salir del programa")
print("=========================================================")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    # Efecto espejo / Mirror effect
    frame = cv2.flip(frame, 1)
    frame_guardar = frame.copy() # Copia limpia sin textos de interfaz / Clean copy without UI text

    # Procesamiento con MediaPipe / Processing with MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect(mp_image)

    # Validar si se detectó al menos una mano / Validate if at least one hand is detected
    mano_detectada = len(resultado.hand_landmarks) > 0

    if mano_detectada:
        h, w, _ = frame.shape
        hand = resultado.hand_landmarks[0]
        
        # Dibujar conexiones del esqueleto (Líneas azules) / Draw skeleton connections (Blue lines)
        for a, b in HAND_CONNECTIONS:
            p1 = hand[a]
            p2 = hand[b]
            cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (255, 0, 0), 2)
        
        # Dibujar los puntos clave (Círculos verdes) / Draw keypoints (Green circles)
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
            
        estado = "MANO DETECTADA / HAND DETECTED"
        color = (0, 255, 0)
    else:
        estado = "NO DETECTADA / NOT DETECTED"
        color = (0, 0, 255)

    # Gestionar directorios por clase/letra / Manage directories per class/letter
    carpeta = os.path.join(DATASET_DIR, letra_actual)
    os.makedirs(carpeta, exist_ok=True)
    num = len([f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")])

    # Mostrar interfaz en pantalla / Display on-screen interface
    cv2.putText(frame, f"Letra / Class: {letra_actual}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Fotos / Images: {num}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, estado, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "ESPACIO guarda solo si detecta la mano", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.imshow("Recolector de Dataset LSM / Dataset Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Controles del teclado / Keyboard controls
    if key == 27:  # Tecla ESC / ESC key
        break
    elif key == 32:  # Barra Espaciadora / Spacebar
        if mano_detectada:
            nombre = f"foto_{num + 1}.jpg"
            cv2.imwrite(os.path.join(carpeta, nombre), frame_guardar)
            print(f"Guardada con éxito / Successfully saved: {nombre} en carpeta '{letra_actual}'")
        else:
            print("Acción denegada: No se guardó porque la mano no está detectada. / Action denied: Hand not detected.")
    elif key == ord('0'):
        letra_actual = "none"
        print(f"-> Clase cambiada a / Class changed to: {letra_actual}")
    elif ord('a') <= key <= ord('z'):
        letra_actual = chr(key).upper()
        print(f"-> Clase cambiada a / Class changed to: {letra_actual}")

cap.release()
cv2.destroyAllWindows()
print("Proceso de recolección finalizado. / Collection process finished.")
