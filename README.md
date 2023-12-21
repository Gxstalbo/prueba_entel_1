# Entel Challenge 

Este repositorio contiene dos archivos .ipynb que corresponden a un desafío de análisis de datos y modelado. A continuación, se detalla el contenido de cada archivo:

## 1. First_part.ipynb
En este archivo, encontrarás los códigos que generan las respuestas para la primera sección del desafío. Esta sección se enfoca en responder las preguntas basicamente.

## 2. Second_part.ipynb
Este archivo contiene el código utilizado para generar los modelos necesarios para el desafío. Se han desarrollado dos modelos distintos:

### 2.1. Modelo con base de entrenamiento a nivel de cliente
Este modelo se entrena utilizando datos a nivel de cliente y se enfoca en tener la data a nivel Mobile_number y sacar datos de dos periodos.

### 2.2. Modelo con base de entrenamiento a nivel cliente - mes
En esta sección, se entrena un segundo modelo que toma en cuenta información a nivel de cliente y mes. La razón para esta separación se basa en aumentar la cantidad de registros

### 2.3. Selección del modelo 2.2.
Después de un análisis exhaustivo, hemos seleccionado el Modelo 2.2. debido a su mejor rendimiento. Utilizamos este modelo para obtener predicciones para el próximo mes.

### 2.4. Análisis del SHAP
Para entender cómo se comportan las variables dentro del modelo seleccionado, realizamos un análisis de SHAP (SHapley Additive exPlanations).

### 2.5. Análisis de Perfil
Realizamos un análisis de perfil que nos permite visualizar grupos de probabilidad versus el promedio de variables, lo que proporciona una mejor comprensión de los resultados del modelo.

### 2.6. Cálculo del ROI
Para medir el impacto del modelo, suponemos costos de campañas y revenue por usuario retenido, lo que nos permite calcular el Retorno de la Inversión (ROI).

### 2.7. Respuestas a las preguntas del ejercicio
Finalmente, proporcionamos respuestas a las preguntas planteadas en el desafío.

## Nota sobre la replicación del código
Para replicar los resultados y el análisis presentados en estos notebooks, es necesario contar con los datos específicos del caso. Estos datos  están incluidos en este repositorio ya que el repositorio está en privado.

Si tienes alguna pregunta o sugerencia, no dudes en ponerte en contacto :)

Have a great day!

<!-- TODO Tu rama dev siempre debería estar sobre tu rama productiva. Nunca se deben hacer cambios directos en tu rama productiva (master en este caso) -->

<!-- TODO Opinión controversial: Jupyter Notebook es pésima herramienta para hacer proyectos serios. De hecho, ni para explorar es buena porque guarda todos tus metadatos y tus commits se corrompen. -->

<!-- TODO A grandes rasgos está bien, pero la segunda parte fue muy caótica y sin una estructura clara. Te recomiendo revisar las recomendaciones el los TODOs. Puedes usar la extensión TODO Tree de VSCode para verlas mejor. -->