# Dataset Image Collector Module (LSM Interpreter)

## English

### Overview
This module provides an interactive tool developed in Python and OpenCV designed to build a high-quality dataset for training hand gesture recognition models. To prevent data corruption or garbage inputs, the program integrates Google's **MediaPipe** (`hand_landmarker.task`) to run real-time inference, ensuring that images are only saved when a valid hand structure is successfully detected.

### How It Works & Key Features
1. **Live Camera Feed & Mirroring:** Captures video streams from the webcam and applies a horizontal flip for a natural user interaction.
2. **Real-Time Hand Validation:** Every frame is processed through the local `hand_landmarker.task` model. If the 21 anatomical landmarks of the hand are identified, the system renders the green skeletal joints and blue connection lines on the display.
3. **Dynamic Class Organization:** Automatically manages a root dataset directory, creating subfolders for each sign class (e.g., letters 'A' through 'Z', or a special 'none' category).
4. **Smart Saving Mechanism:** Pressing the **Spacebar** captures the clean, unmodified camera frame only if the hand detection status is active. It automatically counts existing files within the active folder to name new samples sequentially (`foto_1.jpg`, `foto_2.jpg`, etc.).
5. **Keyboard Controls:** 
   - `A - Z`: Dynamically switch the target capture class.
   - `0`: Switch to the 'none' class folder.
   - `Spacebar`: Save image (guarded by hand detection validation).
   - `ESC`: Safely close the application.

### Free Use & Licensing
Developed for academic research, university social service, and open-source assistance. Distributed under the **Apache 2.0 License**, permitting free use, modification, and educational deployment.

---

## Español

### Descripción General
Este módulo proporciona una herramienta interactiva desarrollada en Python y OpenCV diseñada para construir un conjunto de datos de alta calidad para el entrenamiento de modelos de reconocimiento de gestos de manos. Para prevenir la corrupción de datos o el almacenamiento de imágenes basura, el programa integra **MediaPipe** de Google (`hand_landmarker.task`) para ejecutar inferencia en tiempo real, asegurando que las imágenes solo se guarden cuando se detecta exitosamente una estructura de mano válida.

### Cómo Funciona y Características Principales
1. **Transmisión de Cámara en Vivo y Espejo:** Captura flujos de video desde la cámara web y aplica un volteo horizontal para una interacción natural del usuario.
2. **Validación de Mano en Tiempo Real:** Cada fotograma se procesa a través del modelo local `hand_landmarker.task`. Si se identifican los 21 puntos clave anatómicos de la mano, el sistema renderiza las articulaciones esqueléticas verdes y las líneas de conexión azules en la pantalla.
3. **Organización Dinámica de Clases:** Gestiona automáticamente un directorio raíz de dataset, creando subcarpetas para cada clase de seña (por ejemplo, letras de la 'A' a la 'Z', o una categoría especial 'none').
4. **Mecanismo de Guardado Inteligente:** Al presionar la **Barra Espaciadora**, se captura el fotograma de la cámara limpio y sin modificaciones únicamente si el estado de detección de mano está activo. Cuenta automáticamente los archivos existentes dentro de la carpeta activa para nombrar las nuevas muestras de forma secuencial (`foto_1.jpg`, `foto_2.jpg`, etc.).
5. **Controles de Teclado:**
   - `A - Z`: Cambiar dinámicamente la clase de captura objetivo.
   - `0`: Cambiar a la carpeta de la clase 'none'.
   - `Barra Espaciadora`: Guardar imagen (protegido por la validación de detección de mano).
   - `ESC`: Cerrar la aplicación de forma segura.

### Libre Uso y Licenciamiento
Desarrollado para investigación académica, servicio social universitario y apoyo tecnológico de código abierto. Distribuido bajo la **Licencia Apache 2.0**, permitiendo su uso libre, modificación y despliegue educativo.
