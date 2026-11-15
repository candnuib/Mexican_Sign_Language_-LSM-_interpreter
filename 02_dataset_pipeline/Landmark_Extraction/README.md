# Landmark Extraction Tool (LSM Interpreter)

## English

### Overview
This program automates the batch processing of a structured image dataset, converting visual hand gestures into clean, organized numerical data. It uses Google's MediaPipe (`hand_landmarker.task`) to detect hand structures in stored images and exports their spatial coordinates into structured `.csv` files.

### How It Works
1. **Directory Scanning:** The script automatically traverses the dataset folder, reading subdirectories categorized by sign classes (letters).
2. **Image Loading & Validation:** Each image is loaded via OpenCV and converted to RGB color space. The script checks if MediaPipe successfully detects a hand; if an image fails or contains no detectable hand, it is skipped and logged to maintain data integrity.
3. **Coordinate Extraction:** For every successfully detected hand, the script extracts the 3D coordinates ($X, Y, Z $) of the **21 anatomical keypoints** (fingertips, joints, and wrist base).
4. **Tabular CSV Export:** The extracted data is compiled and saved into individual `.csv` files for each letter (e.g., `A.csv`, `B.csv`). Each row stores the image filename, the target letter, and the 63 corresponding coordinate values ($21 \text{ landmarks} \times 3$ axes).

### Features
* Automated batch processing with progress feedback via `tqdm`.
* Robust validation to filter out corrupted or unreadable images.
* Relative path configuration for seamless execution across different environments and computers.

---

## Español

### Descripción General
Este programa automatiza el procesamiento por lotes de un conjunto de datos de imágenes estructuradas, convirtiendo gestos visuales de manos en datos numéricos limpios y organizados. Utiliza el modelo `hand_landmarker.task` de MediaPipe de Google para detectar estructuras de manos en imágenes almacenadas y exportar sus coordenadas espaciales en archivos `.csv` estructurados.

### Cómo Funciona
1. **Escaneo de Directorios:** El script recorre automáticamente la carpeta del dataset, leyendo subdirectorios categorizados por clases de señas (letras).
2. **Carga y Validación de Imágenes:** Cada imagen se carga mediante OpenCV y se convierte al espacio de color RGB. El script verifica si MediaPipe detecta exitosamente una mano; si una imagen falla o no contiene una mano detectable, se omite y se registra para mantener la integridad de los datos.
3. **Extracción de Coordenadas:** Por cada mano detectada con éxito, el script extrae las coordenadas tridimensionales ($X, Y, Z$) de los **21 puntos clave anatómicos** (yemas de los dedos, articulaciones y base de la muñeca).
4. **Exportación Tabular CSV:** Los datos extraídos se compilan y guardan en archivos `.csv` individuales para cada letra (por ejemplo, `A.csv`, `B.csv`). Cada fila almacena el nombre del archivo de imagen, la letra objetivo y los 63 valores de coordenadas correspondientes ($21 \text{ landmarks} \times 3$ ejes).

### Características
* Procesamiento automatizado por lotes con barra de progreso mediante `tqdm`.
* Validación robusta para filtrar imágenes corruptas o ilegibles.
* Configuración de rutas relativas para una ejecución fluida en diferentes entornos y computadoras.
