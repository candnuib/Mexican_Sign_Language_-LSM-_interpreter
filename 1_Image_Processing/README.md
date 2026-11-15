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
