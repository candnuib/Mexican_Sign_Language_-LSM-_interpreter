# Phase 1: Digital Image Processing and Color Segmentation

## 1. Mathematical Foundations
Image processing begins by modeling the physical scene using a matrix and algebraic approach:

* **Color Space Conversion (RGB to HSV):** To minimize the impact of shadows and ambient lighting variations, the standard BGR image is transformed into the **HSV** (Hue, Saturation, and Value) color space. This decouples pure chromatic information from luminance.
* **Logical Thresholding and Binarization:** A boolean indicator mask $B(x,y)$ is defined through the intersection of strict ranges across chromatic and brightness channels:

$$B(x,y) = \mathbb{I}(H_{min} \leq H \leq H_{max}) \land \mathbb{I}(S_{min} \leq S \leq S_{max}) \land \mathbb{I}(V_{min} \leq V \leq V_{max})$$

  This operation evaluates each pixel individually, assigning a logical value of **1 (white)** if it falls within the skin color range, or **0 (black)** if it is discarded as background.
* **Morphological Operations (Set Theory):** To correct high-frequency noise (isolated pixels or small internal holes caused by textures and highlights), opening and closing morphological filters are applied using a structuring kernel.
* **Spatial Moments and Centroid:** The geometric center or center of mass of the hand is calculated using zero-order ($m_{00}$) and first-order ($m_{10}, m_{01}$) spatial moments:

$$\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$$

---

## 2. Programming Workflow
In the Python and OpenCV implementation, the execution flow follows these steps:

1. **Capture and Normalization:** A frame is acquired from the webcam's optical sensor, horizontally flipped (mirror effect), and resized to a standard resolution of $640 \times 480$.
2. **Chromatic Mapping:** `cv2.cvtColor()` is used to convert the frame to HSV, followed by `cv2.inRange()` to evaluate the numerical skin thresholds.
3. **Spatial Filtering:** `cv2.morphologyEx()` is applied with a rectangular kernel to clean the resulting binary mask.
4. **Geometric Analysis:** Using `cv2.findContours()`, external contours are identified, and utilizing `cv2.moments()`, the exact coordinates of the centroid are calculated in real-time to position the visual indicator on the split screen (`np.hstack`).

---

## 3. Results Obtained
* A functional prototype was successfully achieved, displaying the camera's real-time feed with a geometric bounding box and centroid simultaneously alongside its corresponding binary mask.
* The generated binary matrix clearly differentiates pixel values (white vs. black), facilitating the spatial identification of the hand region under controlled lighting conditions.

---

## 4. Limitations of Classical Processing for Real-Time Gesture Detection
Although this phase demonstrates excellent command of classical image processing, **it is neither a viable nor a robust solution for a real-time Mexican Sign Language (LSM) interpreter** due to the following physical and algorithmic constraints:

* **Critical Lighting Sensitivity:** Static color thresholds mean that minimal changes in ambient light (shadows, direct sunlight, or dim lamps) alter the HSV values, causing the segmentation to fail or lose parts of the hand.
* **Skin Tone and Background Dependency:** The binary algorithm searches for "skin colors," generating false positives if walls, furniture, clothing, or the user's face share similar tones within the field of view.
* **Anatomical Incapacity:** Classical processing handles white spots and flat geometric contours, but **it does not understand anatomy**. It is incapable of identifying individual joints, finger bending, or complex hand rotations, which methodologically justifies the need to evolve toward advanced tools based on keypoint extraction (*landmarks*) and machine learning (*MediaPipe* and neural networks).

# Fase 1: Procesamiento Digital de Imágenes y Segmentación por Color

## 1. Fundamentos Matemáticos
El procesamiento de imágenes inicia modelando la escena física mediante un enfoque matricial y algebraico:

* **Conversión de Espacio de Color (RGB a HSV):** Para minimizar el impacto de las sombras y variaciones de iluminación de la luz ambiental, la imagen se transforma del espacio estándar BGR al espacio **HSV** (Matiz, Saturación y Valor). Esto desacopla la información cromática pura de la luminancia.
* **Umbralización Lógica y Binarización:** Se define una función indicatriz o máscara booleana $B(x,y)$ mediante la intersección de rangos estrictos en los canales cromáticos y de brillo:

$$B(x,y) = \mathbb{I}(H_{min} \leq H \leq H_{max}) \land \mathbb{I}(S_{min} \leq S \leq S_{max}) \land \mathbb{I}(V_{min} \leq V \leq V_{max})$$

  Esta operación evalúa cada píxel individualmente, asignando un valor lógico de **1 (blanco)** si pertenece al rango de piel o **0 (negro)** si se descarta como fondo.
* **Operaciones Morfológicas (Teoría de Conjuntos):** Para corregir el ruido de alta frecuencia (píxeles sueltos o pequeños huecos internos provocados por texturas y brillos), se aplican filtros morfológicos de apertura y cierre utilizando un kernel estructurante.
* **Momentos Espaciales y Centroide:** El cálculo del centro geométrico o centro de masa de la mano se obtiene mediante los momentos espaciales de orden cero ($m_{00}$) y orden uno ($m_{10}, m_{01}$):

$$\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$$

---

## 2. Procesos en la Programación
En el script implementado en Python y OpenCV, el flujo de ejecución sigue estas etapas:

1. **Captura y Normalización:** Se adquiere el fotograma del sensor óptico de la cámara web, se invierte horizontalmente (efecto espejo) y se redimensiona a una resolución estándar de $640 \times 480$.
2. **Mapeo Cromático:** Se utiliza `cv2.cvtColor()` para convertir el fotograma a HSV, seguido de `cv2.inRange()` para evaluar los umbrales numéricos de la piel.
3. **Filtrado Espacial:** Se emplea `cv2.morphologyEx()` con un kernel rectangular para limpiar la máscara binaria resultante.
4. **Análisis Geométrico:** Mediante `cv2.findContours()`, se identifican los contornos externos y, utilizando `cv2.moments()`, se calcula en tiempo real la coordenada exacta del centroide para posicionar el indicador visual en la pantalla dividida (`np.hstack`).

---

## 3. Resultados Obtenidos
* Se logró un prototipo funcional que muestra simultáneamente la toma real de la cámara con delimitación geométrica (*Bounding Box*) y centroide, junto con su respectiva máscara binaria en tiempo real.
* La matriz binaria generada diferencia claramente los valores de píxel (blanco vs. negro), facilitando la identificación espacial de la región correspondiente a la mano en condiciones controladas de iluminación.

---

## 4. Limitaciones del Procesamiento Clásico para la Detección de Gestos en Tiempo Real
Aunque esta fase demuestra un dominio excelente del procesamiento clásico de imágenes, **no es una solución viable ni robusta para un intérprete de Lengua de Señas Mexicana (LSM) en tiempo real** debido a las siguientes restricciones físicas y algorítmicas:

* **Sensibilidad Crítica a la Iluminación:** Los umbrales de color son estáticos, por lo que un cambio mínimo en la luz ambiental (sombras, luz solar directa o lámparas tenues) altera los valores HSV, haciendo que la segmentación falle o pierda partes de la mano.
* **Dependencia del Tono de Piel y Fondo:** El algoritmo binario busca "colores de piel", lo que provoca falsos positivos si hay paredes, muebles, ropa o el rostro del usuario con tonalidades similares dentro del campo de visión.
* **Incapacidad Anatómica:** El procesamiento clásico maneja manchas blancas y contornos geométricos planos, pero **no comprende la anatomía**. No es capaz de identificar articulaciones individuales, flexiones de los dedos o rotaciones complejas de la mano, lo cual justifica metodológicamente la necesidad de evolucionar hacia herramientas avanzadas basadas en extracción de puntos clave (*landmarks*) y aprendizaje automático (*MediaPipe* y redes neuronales).
