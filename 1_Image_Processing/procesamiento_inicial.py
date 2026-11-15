import cv2
import numpy as np

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

print("=== SEGMENTACIÓN DE MANO EN TIEMPO REAL (LADO A LADO) ===")
print("Presiona [ESC] para salir del programa.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    # 1. Voltear horizontalmente para efecto espejo y asegurar dimensiones estándar
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    # 2. Conversión a espacio de color HSV (Excelente para aislar tonos de piel ante sombras)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Rangos ajustados para tono de piel en HSV
    # H (Hue / Matiz): 0 a 20 (cubre tonos de piel desde rojizos hasta amarillentos/canela)
    # S (Saturation / Saturación): 20 a 240 (evita blancos puros como camisas o fondos grises)
    # V (Value / Brillo): 40 a 255 (filtra sombras muy oscuras)
    lower_skin = np.array([0, 20, 40], dtype=np.uint8)
    upper_skin = np.array([28, 240, 255], dtype=np.uint8)

    # Crear la máscara binaria
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 4. Operaciones morfológicas (Filtros espaciales para limpiar ruido de la camisa/fondo)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Elimina puntos sueltos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Rellena huecos internos en la piel

    # 5. Detección de contornos y centroide (Centro de masa)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontrar el contorno de mayor área (asumiendo que es tu mano o rostro)
        c_max = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(c_max) > 1000: # Filtrar ruido pequeño
            # Delimitación geométrica (Bounding Box)
            x, y, w, h = cv2.boundingRect(c_max)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cálculo de momentos espaciales para el centroide (Centro de masa)
            M = cv2.moments(c_max)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Dibujar centroide en la toma real
                cv2.circle(frame, (cX, cY), 6, (0, 0, 255), -1)
                cv2.putText(frame, "Centroide", (cX - 30, cY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 6. Convertir la máscara de 1 canal (grises) a 3 canales (BGR) para poder unirla visualmente
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 7. Unir ambas vistas horizontalmente (Cámara a la izquierda, Filtro a la derecha)
    pantalla_dividida = np.hstack((frame, mask_bgr))

    # Añadir etiquetas de texto descriptivas en pantalla
    cv2.putText(pantalla_dividida, "Toma Real + Centroide", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(pantalla_dividida, "Mascara Binaria (Filtro HSV)", (660, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar la ventana combinada
    cv2.imshow("Investigacion LSM - Procesamiento en Tiempo Real", pantalla_dividida)

    # Salir presionando la tecla [ESC]
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
