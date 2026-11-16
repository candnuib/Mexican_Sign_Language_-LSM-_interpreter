# Real-Time Testing and Validation Module (LSM Interpreter)

## Overview / Descripción General
**English:**
This folder and script (`test_model.py`) correspond to the final stage of the pipeline for the Mexican Sign Language (LSM) interpreter project. Its purpose is to integrate all previously developed components to perform real-time inference and recognition of static signs using the computer's webcam. The program combines computer vision, geometric feature extraction, and the trained Deep Learning model to translate alphabet gestures smoothly in front of the user.

**Español:**
Esta carpeta y script (`test_model.py`) corresponden a la etapa final del pipeline del proyecto de Lengua de Señas Mexicana (LSM). Su propósito es integrar todos los componentes desarrollados previamente para realizar la inferencia y el reconocimiento de señas estáticas en tiempo real utilizando la cámara web de la computadora. El programa une la visión por computadora, la extracción geométrica de características y el modelo de Deep Learning entrenado para traducir los gestos del abecedario de manera fluida frente al usuario.

---

## How Does the Inference Flow Work? / ¿Cómo Funciona el Flujo de Inferencia?

**English:**
The script executes a continuous frame-by-frame loop that performs the following technical steps:
1. **Video Capture and Mirroring:** OpenCV captures the live webcam stream and applies a horizontal flip (mirror effect) to facilitate natural user interaction.
2. **Landmark Detection (MediaPipe):** Each frame in RGB format is analyzed by the local model `hand_landmarker.task`. If a hand is detected, the system localizes the 21 anatomical key points (joints and fingertips) with high precision.
3. **Input Vector Generation:** The three-dimensional coordinates ($X, Y, Z$) of the 21 landmarks are flattened into a single continuous list, forming the 63-variable input vector expected by the neural network.
4. **Prediction with the Neural Network (TensorFlow / Keras):** The 63-feature vector is fed into the trained model (`static_letters.keras`). The network processes the information through its dense layers and outputs a probability distribution for each class.
5. **Interpretation and Visual Rendering:** 
   * `np.argmax()` is used to identify the letter with the highest probability, and its confidence or accuracy percentage is calculated.
   * The predicted letter text (e.g., *A (95.4%)*) and the geometric hand skeleton (key points and connections) are drawn on the screen in real-time to visually verify that the system is operating correctly.

**Español:**
El script ejecuta un ciclo continuo fotograma por fotograma que realiza los siguientes pasos técnicos:
1. **Captura y Espejo de Video:** OpenCV captura la transmisión en vivo de la cámara web y aplica un giro horizontal (efecto espejo) para facilitar la interacción natural del usuario.
2. **Detección de Landmarks (MediaPipe):** Cada fotograma en formato RGB es analizado por el modelo local `hand_landmarker.task`. Si se detecta una mano, el sistema localiza con alta precisión los 21 puntos clave anatómicos (articulaciones y yemas de los dedos).
3. **Generación del Vector de Entrada:** Las coordenadas tridimensionales ($X, Y, Z$) de los 21 landmarks se aplanan en una sola lista continua, conformando el vector de 63 variables de entrada que la red neuronal espera.
4. **Predicción con la Red Neuronal (TensorFlow / Keras):** El vector de 63 características se introduce al modelo entrenado (`static_letters.keras`). La red procesa la información a través de sus capas densas y emite una distribución de probabilidades para cada clase.
5. **Interpretación y Renderizado Visual:** 
   * Se utiliza `np.argmax()` para identificar la letra con mayor probabilidad y se calcula su porcentaje de confianza o acierto.
   * Se dibuja en tiempo real sobre la pantalla el texto con la letra predicha (ej. *A (95.4%)*) y el esqueleto geométrico de la mano (puntos clave y conexiones) para verificar visualmente que el sistema está operando correctamente.

---

## Requirements and File Configuration / Requisitos y Configuración de Archivos

**English:**
For this script to function optimally, make sure the following files are located within your project structure (pointing correctly via relative paths):
* `../modelos/hand_landmarker.task`: Official MediaPipe model for hand detection.
* `../modelos/static_letters.keras`: The binary file containing the weights and architecture of the already-trained neural network.
* `../modelos/labels.txt`: The label dictionary that translates the model's numerical outputs back into alphabet letters (e.g., `0 -> A`).

**Español:**
Para que este script funcione de manera óptima, asegúrate de tener ubicados los siguientes archivos dentro de la estructura de tu proyecto (apuntando correctamente mediante rutas relativas):
* `../modelos/hand_landmarker.task`: Modelo oficial de MediaPipe para la detección de la mano.
* `../modelos/static_letters.keras`: El archivo binario con los pesos y la arquitectura de la red neuronal ya entrenada.
* `../modelos/labels.txt`: El diccionario de etiquetas que traduce las salidas numéricas del modelo de nuevo a letras del abecedario (ej. `0 -> A`).

---

## Program Execution / Ejecución del Programa

**English:**
With your virtual environment active and dependencies installed (`opencv-python`, `mediapipe`, `numpy`, `tensorflow`), run the script from your terminal:

```bash
python test_model.py
