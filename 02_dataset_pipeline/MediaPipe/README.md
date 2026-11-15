# Real-Time Hand Landmark Detection & Dataset Pipeline (LSM Interpreter)

## English

### Overview
This repository section contains the core modules for hand detection, landmark extraction using MediaPipe's `hand_landmarker.task`, and the automated dataset pipeline for the Mexican Sign Language (LSM) electronic interpreter project. This phase bridges real-time computer vision with machine learning by converting visual frames into structured numerical coordinates.

### How It Works
1. **Video Acquisition:** Captures continuous frames using OpenCV from the connected webcam[cite: 4, 5].
2. **Color Space Conversion:** Transforms standard BGR frames into the RGB format required by MediaPipe[cite: 4].
3. **Inference Engine (`hand_landmarker.task`):** Utilizes Google's specialized hand landmarker model to detect hand presence and localize 3D coordinates ($X, Y, Z$) for **21 anatomical keypoints**[cite: 5, 7].
4. **Dataset Pipeline & CSV Export:** Automatically processes categorized image folders (static signs), extracts the 21 landmarks per image, validates detections, and exports the data into structured `.csv` files for each letter[cite: 7, 8, 9].

### Technical Architecture
* **Detection Model:** `hand_landmarker.task` (edge computing-optimized model executing palm detection and landmark regression).
* **Data Representation:** Each sample is reduced from a heavy image matrix to a compact tabular row containing 63 coordinate values ($21 \text{ landmarks} \times 3$ axes) plus metadata[cite: 7].
* **Deep Learning Preparation:** The structured CSV dataset is prepared for training neural networks (TensorFlow/Keras) to classify static LSM hand signs reliably[cite: 8].

### Free Use & Licensing
Developed for academic research, university social service, and open-source assistance. Distributed under the **Apache 2.0 License**, permitting free use, modification, and educational deployment.

---

## Español

### Descripción General
Esta sección del repositorio contiene los módulos principales para la detección de manos, la extracción de puntos clave mediante el modelo `hand_landmarker.task` de MediaPipe y el pipeline automatizado de conjuntos de datos para el proyecto del intérprete electrónico de Lengua de Señas Mexicana (LSM). Esta fase conecta la visión por computadora en tiempo real con el aprendizaje automático, convirtiendo fotogramas visuales en coordenadas numéricas estructuradas.

### Cómo Funciona
1. **Adquisición de Video:** Captura fotogramas continuos mediante OpenCV desde la cámara web conectada[cite: 4, 5].
2. **Conversión de Espacio de Color:** Transforma el fotograma estándar BGR al formato RGB requerido por MediaPipe[cite: 4].
3. **Motor de Inferencia (`hand_landmarker.task`):** Utiliza el modelo especializado de Google para detectar la presencia de la mano y localizar las coordenadas tridimensionales ($X, Y, Z$) de **21 puntos clave anatómicos**[cite: 5, 7].
4. **Pipeline de Dataset y Exportación CSV:** Procesa automáticamente carpetas de imágenes categorizadas (señas estáticas), extrae los 21 landmarks por imagen, valida las detecciones y exporta la información en archivos `.csv` estructurados para cada letra[cite: 7, 8, 9].

### Arquitectura Técnica
* **Modelo de Detección:** `hand_landmarker.task` (modelo optimizado para computación en el borde que ejecuta detección de palma y regresión de landmarks).
* **Representación de Datos:** Cada muestra se reduce de una matriz de imagen pesada a una fila tabular compacta que contiene 63 valores de coordenadas ($21 \text{ landmarks} \times 3$ ejes) más metadatos[cite: 7].
* **Preparación para Deep Learning:** El dataset CSV estructurado se prepara para alimentar el entrenamiento de redes neuronales (TensorFlow/Keras) orientadas a clasificar de manera confiable las señas estáticas de la LSM[cite: 8].

### Libre Uso y Licenciamiento
Desarrollado para investigación académica, servicio social universitario y apoyo tecnológico de código abierto. Distribuido bajo la **Licencia Apache 2.0**, permitiendo su uso libre, modificación y despliegue educativo.
