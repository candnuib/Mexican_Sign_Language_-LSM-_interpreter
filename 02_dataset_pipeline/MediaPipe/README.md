# Real-Time Hand Landmark Detection (MediaPipe Phase)

## English

### Overview
This repository section contains the implementation of the hand detection and landmark extraction module using **MediaPipe** and Python[cite: 4, 7]. This phase represents a major technological upgrade from classical color thresholding, providing robust real-time tracking of hand joints regardless of background complexity or skin tone variations.

### How It Works
1. **Video Acquisition:** Captures continuous frames using OpenCV from the connected webcam[cite: 4, 5].
2. **Color Conversion:** Transforms the default BGR frame into an RGB format required by MediaPipe.
3. **Inference Engine:** Loads the pre-trained `gesture_recognizer.task` model to perform edge computing analysis[cite: 4, 8].
4. **Landmark Extraction:** Detects 21 3D anatomical keypoints representing the hand's joints and fingertips[cite: 7].
5. **Visual Rendering:** Maps normalized coordinates back to pixel space and draws the skeletal structure and keypoints directly onto the video feed[cite: 4].

### Training and Model Architecture
The system relies on Google's pre-trained MediaPipe machine learning model (`gesture_recognizer.task`), optimized for lightweight real-time inference on edge devices. It utilizes a two-tier cascade architecture: a palm detector that crops the Region of Interest (ROI), followed by a hand landmark model that predicts precise coordinate vectors ($X, Y, Z$) for each joint.

### Free Use & Licensing
This project is developed for academic research, social service, and open-source assistance. It is distributed under the **Apache 2.0 License**, permitting free use, modification, and educational deployment.

---

## Español

### Descripción General
Esta sección del repositorio contiene la implementación del módulo de detección de manos y extracción de puntos clave (*landmarks*) utilizando **MediaPipe** y Python[cite: 4, 7]. Esta fase representa una evolución tecnológica superior frente al filtrado clásico por color, ofreciendo un seguimiento robusto en tiempo real de las articulaciones de la mano sin verse afectada por la complejidad del fondo o las variaciones en el tono de piel.

### Cómo Funciona
1. **Adquisición de Video:** Captura fotogramas continuos mediante OpenCV desde la cámara web conectada[cite: 4, 5].
2. **Conversión de Color:** Transforma el fotograma estándar BGR al formato RGB requerido por MediaPipe.
3. **Motor de Inferencia:** Carga el modelo pre-entrenado `gesture_recognizer.task` para realizar análisis mediante computación en el borde (*edge computing*)[cite: 4, 8].
4. **Extracción de Landmarks:** Detecta 21 puntos clave anatómicos tridimensionales que representan las articulaciones y yemas de los dedos[cite: 7].
5. **Renderizado Visual:** Mapea las coordenadas normalizadas al espacio de píxeles y dibuja la estructura esquelética y los puntos directamente sobre la transmisión de video[cite: 4].

### Entrenamiento y Arquitectura del Modelo
El sistema se basa en el modelo de aprendizaje automático pre-entrenado de Google (`gesture_recognizer.task`), optimizado para inferencia ligera en tiempo real sobre dispositivos periféricos. Utiliza una arquitectura en cascada de dos etapas: un detector de palma que recorta la Región de Interés (ROI), seguido de un modelo de puntos clave que predice vectores de coordenadas precisas ($X, Y, Z$) para cada articulación.

### Libre Uso y Licenciamiento
Este proyecto ha sido desarrollado con fines de investigación académica, servicio social universitario y apoyo tecnológico de código abierto. Se distribuye bajo la **Licencia Apache 2.0**, permitiendo su uso libre, modificación y despliegue educativo.
