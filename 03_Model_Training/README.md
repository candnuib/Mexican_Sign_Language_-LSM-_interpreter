# Módulo de Entrenamiento de Redes Neuronales (LSM Interpreter)

## Descripción General y Ruta del Proyecto
Este módulo representa el núcleo de aprendizaje automático de la investigación para el intérprete de Lengua de Señas Mexicana (LSM). La ruta metodológica adoptada consistió en pasar de un enfoque basado en reglas matemáticas rígidas a un modelo robusto de **Deep Learning** capaz de clasificar señas estáticas de forma precisa. 

El flujo general implementado abarca desde el procesamiento del dato crudo hasta la evaluación final del modelo:
1. **Recolección de Datos (image_colector):** Captura fotográfica controlada de la mano variando distancia y ángulos para evitar sobreajuste a una sola postura.
2. **Extracción y Limpieza de Landmarks (landmark_extraction):** Uso de MediaPipe para aislar los 21 puntos clave de la mano, transformando cada imagen pesada en un vector geométrico compacto de **63 coordenadas** (X, Y, Z), y filtrando fotogramas basura o detecciones fallidas.
3. **Consolidación y Validación Tabular:** Integración de los archivos CSV individuales por cada letra en un único dataset estructurado.
4. **Partición de Datos:** División estratificada para entrenamiento, validación y prueba.
5. **Entrenamiento Supervisado (TensorFlow/Keras):** Optimización de una red neuronal densa (Multilayer Perceptron).

---

## Preparación y División del Dataset
Para garantizar que el modelo aprenda a generalizar y no a simplemente "memorizar" las fotos, el conjunto total de datos (compuesto por **2,100 muestras** distribuidas equitativamente en **21 letras estáticas**) se dividió de la siguiente manera:

* **Conjunto de Entrenamiento (80% - 1,680 muestras):** Datos con los que la red neuronal ajusta sus pesos internos y aprende las relaciones geométricas de las señas.
* **Conjunto de Validación (10% - 210 muestras):** Datos utilizados durante el entrenamiento para medir el rendimiento en tiempo real y aplicar técnicas de control como *Early Stopping* y *ModelCheckpoint* (guardar el mejor modelo sin sobreajustar).
* **Conjunto de Prueba / Test (10% - 210 muestras):** Datos completamente aislados que el modelo **jamás ve** durante el aprendizaje. Se emplean al final para evaluar de manera honesta y real qué tan bien predice datos nuevos.

---

## Arquitectura de la Red Neuronal (TensorFlow / Keras)
El script `entrenar_modelo.py` implementa una red neuronal secuencial (*Sequential*) optimizada para procesar los datos tabulares de los landmarks:

* **Capa de Entrada (Input Layer):** Recibe un vector de **63 valores** de entrada (correspondientes a las coordenadas X, Y, Z de los 21 puntos clave de la mano).
* **Primera Capa Oculta (Dense - 128 neuronas):** Capa densamente conectada con función de activación `ReLU` encargada de extraer patrones complejos y relaciones espaciales entre las articulaciones de los dedos.
* **Segunda Capa Oculta (Dense - 64 neuronas):** Capa de abstracción superior que reduce la dimensionalidad y consolida las características geométricas de la seña.
* **Capa de Salida (Output Layer - 21 neuronas):** Capa final con activación `Softmax` que entrega una distribución de probabilidad para cada una de las **21 clases o letras estáticas** del abecedario evaluado.

---

## Conceptos Clave Explicados

### 1. ¿Qué es la función de activación ReLU (Rectified Linear Unit)?
En una red neuronal, las neuronas realizan sumas ponderadas de sus entradas. Sin una función de activación, la red se comportaría siempre como una simple línea recta (operación lineal), sin importar cuántas capas tenga. 
* **Lo que hace ReLU:** Es una función matemática que toma cualquier valor numérico y **elimina los números negativos convirtiéndolos en 0**, mientras que deja pasar los números positivos intactos ($f(x) = \max(0, x)$).
* **Por qué se usa:** Aporta **no-linealidad** a la red. Esto le permite a tu modelo aprender patrones sumamente complejos y combinaciones de dobleces en los dedos que una línea recta jamás podría capturar, haciéndolo de forma computacionalmente muy eficiente y rápida.

### 2. ¿Qué hace el optimizador Adam (Adaptive Moment Estimation)?
Durante el entrenamiento, la red neuronal comete errores y necesita ajustar sus "pesos" (la importancia que le da a cada coordenada) para equivocarse cada vez menos. El optimizador es el algoritmo encargado de hacer esta corrección.
* **Lo que hace Adam:** Es uno de los optimizadores más potentes y populares en Deep Learning. Combina lo mejor de dos mundos:
  * Ajusta la velocidad de aprendizaje (*learning rate*) de manera individual para cada parámetro de la red.
  * Utiliza "momento" (como una bola de nieve rodando cuesta abajo) para acelerar en la dirección correcta y superar pequeños baches o valles superficiales donde el modelo podría quedarse atrapado (*mínimos locales*).
* **Por qué se usa:** Hace que el entrenamiento sea sumamente estable, converja mucho más rápido y requiera menos ajustes manuales complejos de hiperparámetros.

---

## Resultados Obtenidos y su Significado

Al finalizar el proceso de entrenamiento y evaluar el modelo contra el conjunto de prueba aislado, se obtuvieron las siguientes métricas:

* **Accuracy (Precisión):** **`88.10%`**
* **Loss (Pérdida):** **`0.3184`**

### ¿Qué significan estos resultados?
* **El Accuracy (88.10%):** Indica que de cada 100 señas realizadas frente al sistema en el conjunto de prueba independiente, el modelo es capaz de identificar y clasificar correctamente la letra de la seña en aproximadamente **88 ocasiones**. Para una primera fase experimental sin calibraciones complejas de hardware, es un porcentaje de efectividad alto y muy sólido que demuestra la viabilidad técnica del proyecto.
* **El Loss (0.3184):** Es la medida numérica del error que calcula la función de pérdida (*sparse_categorical_crossentropy*). Un valor bajo (cercano a 0) significa que las predicciones probabilísticas de la red coinciden estrechamente con las etiquetas reales de las letras. Refleja que el modelo no solo adivina, sino que tiene una alta certeza matemática en sus decisiones.

---

### Archivos Generados
Como salida de este script, se almacenan los siguientes artefactos en la carpeta de modelos:
1. **`letras_estaticas.keras`:** El archivo binario que contiene la estructura de la red y los pesos neuronales ya entrenados ("la inteligencia adquirida").
2. **`labels.txt`:** El diccionario que mapea las salidas numéricas de la red con su respectiva letra del abecedario (ej. `0 -> A`, `1 -> B`, etc.).

<img width="648" height="363" alt="image" src="https://github.com/user-attachments/assets/8b5c59d9-1c39-400b-a528-b8d37a3872ec" />

