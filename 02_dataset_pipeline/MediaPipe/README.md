# Real-Time Hand Detection & Landmark Visualization

<img width="1543" height="538" alt="image" src="https://github.com/user-attachments/assets/4be986ed-ffca-4ac7-83d9-8ded8872774e" />

## English

### Overview
This module demonstrates the practical application of real-time computer vision using **MediaPipe** (`hand_landmarker.task`) to detect hands through a live webcam feed and render their anatomical structure directly on the screen.

### How It Works & Key Features
1. **Video Capture & Mirroring:** OpenCV captures continuous frames from the webcam and flips them horizontally to provide a natural mirror view for the user.
2. **Color Space Conversion:** Standard BGR frames are converted into RGB format, which is the required input format for MediaPipe's inference pipeline.
3. **Edge Computing Inference:** The system loads Google's pre-trained `hand_landmarker.task` model to execute lightweight machine learning locally on the device.
4. **Landmark Extraction & Skeletal Rendering:** 
   - The model detects the hand region of interest (ROI) and predicts the exact 3D coordinates ($X, Y, Z$) of **21 anatomical keypoints** (fingertips, joints, and wrist base).
   - The script iterates through these normalized coordinates, scales them to the frame's pixel dimensions, and draws green circular markers for each point.
   - Using predefined skeletal connection arrays (`HAND_CONNECTIONS`), blue lines are dynamically drawn between related keypoints to form a complete, recognizable hand skeleton in real-time.

### Origin, Training, and Free Use
* **Developed by Google:** MediaPipe is an open-source framework built and maintained by Google, specifically optimized for cross-platform machine learning solutions in real-time streaming media.
* **How It Was Trained:** The underlying model (`hand_landmarker.task`) was trained by Google on massive, diverse datasets of hand images annotated with precise 3D joints, enabling it to learn robust features invariant to skin tones, lighting conditions, or background clutter.
* **Open Source & Licensing:** Distributed under the **Apache 2.0 License**, the tool is entirely free to use, modify, and distribute for academic research, university projects, and commercial applications.

---

## Español

### Descripción General
Este módulo demuestra la aplicación práctica de la visión por computadora en tiempo real utilizando **MediaPipe** (`hand_landmarker.task`) para detectar manos a través de una transmisión de cámara web en vivo y renderizar su estructura anatómica directamente en la pantalla.

### Cómo Funciona y Características Principales
1. **Captura de Video y Efecto Espejo:** OpenCV captura fotogramas continuos desde la cámara web y los invierte horizontalmente para ofrecer una vista de espejo natural al usuario.
2. **Conversión de Espacio de Color:** Los fotogramas estándar en BGR se convierten al formato RGB, que es el espacio de entrada requerido por el pipeline de inferencia de MediaPipe.
3. **Inferencia por Computación en el Borde (*Edge Computing*):** El sistema carga el modelo pre-entrenado `hand_landmarker.task` de Google para ejecutar aprendizaje automático ligero de manera local en el dispositivo.
4. **Extracción de Landmarks y Renderizado Esquelético:** 
   - El modelo detecta la región de interés (ROI) de la mano y predice las coordenadas tridimensionales exactas ($X, Y, Z$) de **21 puntos clave anatómicos** (yemas de los dedos, articulaciones y base de la muñeca).
   - El script recorre estas coordenadas normalizadas, las escala a las dimensiones en píxeles del fotograma y dibuja marcadores circulares verdes para cada punto.
   - Utilizando arreglos de conexión esquelética predefinidos (`HAND_CONNECTIONS`), se dibujan líneas azules de manera dinámica entre los puntos clave relacionados para formar un esqueleto de la mano completo y reconocible en tiempo real.

### Origen, Entrenamiento y Libre Uso
* **Desarrollado por Google:** MediaPipe es un framework de código abierto creado y mantenido por Google, optimizado específicamente para soluciones de aprendizaje automático multiplataforma en tiempo real.
* **Cómo Fue Entrenado:** El modelo subyacente (`hand_landmarker.task`) fue entrenado por Google utilizando conjuntos de datos masivos y diversos de imágenes de manos anotadas con articulaciones tridimensionales precisas, lo que le permite aprender características robustas independientes del tono de piel, las condiciones de iluminación o la complejidad del fondo.
* **Código Abierto y Licenciamiento:** Distribuido bajo la **Licencia Apache 2.0**, la herramienta es de libre uso, modificación y distribución para investigación académica, proyectos universitarios y aplicaciones comerciales.
