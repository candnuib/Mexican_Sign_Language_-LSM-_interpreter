# Dataset Pipeline: Collection, Validation & Landmark Extraction (LSM Interpreter)

## English

### Overview
The `dataset_pipeline` folder houses the foundational workflow required to bridge real-time computer vision with Deep Learning for the Mexican Sign Language (LSM) interpreter project. Instead of feeding heavy, noisy raw image tensors directly into a neural network, this pipeline transforms visual hand gestures into clean, structured numerical datasets optimized for efficient machine learning training.

### Why Structured Coordinates Over Raw Pixels?
* **Massive Dimensionality Reduction:** A standard color frame contains nearly 921,600 individual values ($640 \times 480 \times 3$ channels). Extracting hand landmarks reduces this to just **63 numerical values** ($21 \text{ anatomical joints} \times 3$ spatial coordinates: $X, Y, Z$), lowering computational complexity exponentially.
* **Invariance to Background and Lighting:** Training on raw pixels makes models highly vulnerable to background clutter, changing ambient light, and diverse skin tones. Using geometric coordinates focuses exclusively on the relative spatial proportions and joint orientations of the hand, yielding a vastly more robust classifier.
* **Faster Convergence and Higher Accuracy:** Neural networks train significantly faster and achieve higher generalization when supplied with normalized, low-dimensional coordinate spaces.

### Pipeline Components & Tools

1. **Dataset Collector (`recolector.py`)**
   * **Purpose:** Automates the acquisition of static sign images using OpenCV and the computer webcam.
   * **Key Features:** It integrates Google's MediaPipe (`hand_landmarker.task`) in real-time. The script validates that a human hand is actively present before allowing the user to capture an image, preventing garbage data or empty frames. It automatically manages class subfolders (letters 'A' through 'Z', and 'none'), names files sequentially (`foto_1.jpg`, `foto_2.jpg`, etc.), and features a live on-screen UI displaying the active class and total sample count.

2. **Dataset Inspector / Validator (`revisar_dataset.py`)**
   * **Purpose:** Conducts quality assurance across the captured image repository prior to model training.
   * **Key Features:** Iterates through every folder to verify that MediaPipe can successfully detect hand landmarks in all stored samples, catching edge cases where lighting or awkward hand angles may have compromised detection.

3. **Landmark Extraction (`extraer_landmarks.py`)**
   * **Purpose:** Converts the validated image dataset into tabular training data.
   * **Key Features:** Processes each image in batch, extracts the 3D coordinates of the 21 keypoints, and exports them into structured `.csv` files (one per class/letter), pairing each sample with its filename and target label.

### Free Use & Licensing
Developed for academic research, university social service, and open-source assistance. Distributed under the **Apache 2.0 License**, permitting free use, modification, and educational deployment.

---

## Español

### Descripción General
La carpeta `dataset_pipeline` alberga el flujo de trabajo fundamental necesario para conectar la visión por computadora en tiempo real con el aprendizaje profundo (*Deep Learning*) para el proyecto del intérprete de Lengua de Señas Mexicana (LSM). En lugar de alimentar una red neuronal directamente con tensores de imágenes crudas y pesadas, este pipeline transforma los gestos visuales de las manos en conjuntos de datos numéricos limpios y estructurados, optimizados para un entrenamiento eficiente de Machine Learning.

### ¿Por qué Coordenadas Estructuradas en lugar de Píxeles Crudos?
* **Reducción Masiva de Dimensionalidad:** Un fotograma de color estándar contiene cerca de 921,600 valores individuales ($640 \times 480 \times 3$ canales). Extraer los landmarks de la mano reduce esto a tan solo **63 valores numéricos** ($21 \text{ articulaciones anatómicas} \times 3$ coordenadas espaciales: $X, Y, Z$), disminuyendo la complejidad computacional de forma exponencial.
* **Invariancia al Fondo y la Iluminación:** Entrenar con píxeles crudos hace que los modelos sean altamente vulnerables al ruido del fondo, cambios de luz ambiental y diversos tonos de piel. Utilizar coordenadas geométricas se centra exclusivamente en las proporciones espaciales relativas y orientaciones articulares de la mano, logrando un clasificador notablemente más robusto.
* **Convergencia más Rápida y Mayor Precisión:** Las redes neuronales entrenan de manera significativamente más rápida y alcanzan una mejor generalización cuando se alimentan de espacios de coordenadas normalizados y de baja dimensión.

### Componentes y Herramientas del Pipeline

1. **Recolector de Dataset (`recolector.py`)**
   * **Propósito:** Automatiza la adquisición de imágenes de señas estáticas utilizando OpenCV y la cámara web de la computadora.
   * **Características Principales:** Integra en tiempo real MediaPipe de Google (`hand_landmarker.task`). El script valida que una mano humana esté presente activamente antes de permitir que el usuario capture una imagen, evitando datos basura o fotogramas vacíos. Gestiona automáticamente las subcarpetas por clases (letras de la 'A' a la 'Z', y 'none'), nombra los archivos de manera secuencial (`foto_1.jpg`, `foto_2.jpg`, etc.) y cuenta con una interfaz visual en pantalla que muestra la clase activa y el conteo total de muestras.

2. **Inspector / Validador de Dataset (`revisar_dataset.py`)**
   * **Propósito:** Realiza un control de calidad en todo el repositorio de imágenes capturadas antes de entrenar el modelo.
   * **Características Principales:** Itera a través de cada carpeta para verificar que MediaPipe pueda detectar con éxito los landmarks de la mano en todas las muestras almacenadas, detectando casos límite donde la iluminación o posturas difíciles de la mano hayan comprometido la detección.

3. **Extracción de Landmarks (`extraer_landmarks.py`)**
   * **Propósito:** Convierte el dataset de imágenes validadas en datos tabulares de entrenamiento.
   * **Características Principales:** Procesa cada imagen por lotes, extrae las coordenadas 3D de los 21 puntos clave y las exporta en archivos `.csv` estructurados (uno por cada clase/letra), asociando cada muestra con su nombre de archivo y etiqueta objetivo.

### Libre Uso y Licenciamiento
Desarrollado para investigación académica, servicio social universitario y apoyo tecnológico de código abierto. Distribuido bajo la **Licencia Apache 2.0**, permitiendo su uso libre, modificación y despliegue educativo.
