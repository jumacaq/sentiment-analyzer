# Análisis de sentimientos de tweets

## Descripción general
Este proyecto clasifica el sentimiento de los tweets como **positivo o negativo** utilizando un conjunto de datos de **1.600.000 tweets preprocesados**. Extrae características lingüísticas y estadísticas clave y luego entrena un modelo de aprendizaje automático para la predicción de sentimientos, finalmente el modelo se desplegará como aplicación en streamlit.

## Tecnologías utilizadas
Este proyecto integra varias bibliotecas y herramientas clave:

- **[pandas](https://pandas.pydata.org/)**: manipulación y análisis de datos.
- **[nltk](https://www.nltk.org/)**: procesamiento de lenguaje natural (eliminación de palabras vacías, tokenización).
- **[VADER Sentiment](https://github.com/cjhutto/vaderSentiment)**: herramienta de análisis de sentimientos basada en reglas para texto de redes sociales.
- **[TextBlob](https://textblob.readthedocs.io/en/dev/)**: análisis de sentimientos y preprocesamiento de texto.
- **[scikit-learn](https://scikit-learn.org/)** – Entrenamiento y evaluación de modelos de aprendizaje automático.
- **[matplotlib](https://matplotlib.org/)** y **[seaborn](https://seaborn.pydata.org/)** – Visualización de datos.
- **[emoji](https://pypi.org/project/emoji/)** – Maneja emojis en el procesamiento de texto.
- **[Expresiones regulares (re)](https://docs.python.org/3/library/re.html)** – Extrae patrones del texto.

## Instalación
Instalar dependencias requeridas:
```sh
pip install pandas regex string matplotlib seaborn scikit-learn nltk emoji vaderSentiment textblob
```

## Procesamiento de datos
Durante el preprocesamiento, se realizan las siguientes operaciones clave:

- **Eliminación de URL, menciones y emojis**: Se eliminan del texto ya que pueden agregar ruido al análisis.
- **Eliminación de signos de puntuación y caracteres especiales**: Para normalizar el texto antes de aplicar transformaciones.
- **Transformación a minúsculas**: Para evitar que las palabras en mayúsculas se traten como palabras diferentes.
- **Generación de características adicionales**: Se generan nuevas variables en función del texto, como longitud del tweet, conteo de palabras vacías, densidad de palabras en mayúsculas, entre otras.

## Generación de características
Se generaron varias características a partir del texto de los tweets. Estas incluyen:

- Longitud del tweet (`tweet_length`).
- Recuento de emojis (`emoji_count`).
- Recuento de signos de exclamación/interrogación (`exclamation_count`, `question_count`).
- Densidad de palabras en mayúsculas (`capital_word_density`).
- Recuento de palabras (`word_count`).
- Recuento de palabras vacías (`stopword_count`).
- Recuento de palabras únicas (`unique_word_count`).
- Proporción de palabras repetidas (`repeated_word_proportion`).
- Subjetividad de sentimientos mediante TextBlob (`textblob_subjectivity`).
- Recuento de puntuación (`punctuation_count`).
- Recuento de menciones y hashtags (`mention_count`, `hashtag_count`).
- Detección de sarcasmo mediante VADER y análisis de texto (`sarcasm`).

## División de datos
El conjunto de datos se dividió en conjuntos de entrenamiento, validación y prueba utilizando una proporción del 70 % para el entrenamiento, el 30% para la prueba.

```
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Ahora dividir X_temp en test (50%) y validation (50%)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

```

## Modelado
El conjunto de datos se dividió en conjuntos de entrenamiento, prueba y validación, y luego se entrenaron los modelos de aprendizaje automático (**XGBoost** , **LightGBM** ) utilizando las características generadas a partir de los textos procesados.

## Evaluación del modelo
Finalmente, el modelo se evaluó utilizando métricas de clasificación como **AUC-ROC, exactitud, precisión, recuperación y puntaje F1**. Se realizó un ajuste adicional de hiperparámetros utilizando técnicas como **HalvingGridSearchCV** para los modelos **XGBoost** y **LightGBM** 

## Conclusiones
### Análisis de selección de modelos

Según las métricas proporcionadas, **LightGBM** sería el mejor modelo para implementar por las siguientes razones:

1. **Puntaje AUC-ROC superior**: LightGBM logra un AUC-ROC de 0,8222 en comparación con el 0,7443 de XGBoost. Esta es una mejora sustancial (aproximadamente un 10,5 % más) e indica que LightGBM tiene una capacidad de discriminación significativamente mejor entre las clases en varios umbrales.

2. **Coeficiente de Gini más alto**: el coeficiente de Gini de LightGBM (0,6443) es mucho mejor que el de XGBoost (0,4886). El coeficiente de Gini, derivado del AUC-ROC (Gini = 2*AUC-1), confirma la capacidad de clasificación superior de LightGBM.

3. **Métricas de clasificación similares**: ambos modelos funcionan de manera casi idéntica en precisión, recuperación, puntuación F1 y exactitud (aproximadamente 0,74), y XGBoost tiene una ventaja insignificante en precisión (0,7444 frente a 0,7429).

4. **Análisis de la matriz de confusión**:
- XGBoost tiene un poco menos de falsos negativos (27.788 frente a 28.486)
- LightGBM tiene un poco menos de falsos positivos (33.228 frente a 33.558)
- Las diferencias en las matrices de confusión son mínimas y no compensan la importante ventaja de AUC-ROC

Si bien XGBoost tiene una ventaja marginal en precisión (0,0015 más), esta pequeña diferencia se ve eclipsada por la capacidad sustancialmente mejor de LightGBM para clasificar las predicciones correctamente, como lo demuestran las métricas AUC-ROC y Gini.

El AUC-ROC es particularmente importante si:
- El umbral de clasificación óptimo puede cambiar en el futuro
- Los costos de los falsos positivos y los falsos negativos son diferentes o pueden cambiar
- Necesita estimaciones de probabilidad en lugar de solo predicciones binarias

Además, LightGBM es generalmente conocido por sus velocidades de entrenamiento más rápidas y su menor uso de memoria en comparación con XGBoost, lo que podría ser ventajoso en entornos de producción, aunque esto no se midió directamente en las métricas proporcionadas.

Por lo tanto, LightGBM sería el modelo recomendado para implementar en función de estas métricas de evaluación.

## Despliegue en Streamlit
Se despliega el modelo elegido como aplicación web usando la platafroma Streamlit. 

