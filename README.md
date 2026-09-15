# Análisis de desempeño de Random Forest

Este repositorio corresponde a la etapa de **análisis y evaluación del desempeño del modelo** del módulo M2.

Se analiza un modelo de **Random Forest Classifier** implementado con **Scikit-learn** y entrenado sobre el **Digits Dataset**, con el objetivo de estudiar su comportamiento en los conjuntos de training, validation y test, así como diagnosticar su nivel de bias, varianza y ajuste.

---

## Objetivo

El objetivo de esta actividad es evaluar el comportamiento del modelo y determinar si presenta:

- Underfitting
- Ajuste adecuado
- Overfitting
- Bias alto, medio o bajo
- Varianza alta, media o baja

Además, se utiliza una técnica de regularización mediante ajuste de hiperparámetros para mejorar la capacidad de generalización del modelo.

---

## Dataset

Se utilizó el **Digits Dataset** incluido en Scikit-learn.

Características principales:

- 1,797 observaciones
- 64 features
- 10 clases
- Dígitos del `0` al `9`
- Cada observación representa una imagen de `8 × 8` píxeles

---

## División de los datos

El dataset se dividió en tres conjuntos independientes:

| Conjunto | Observaciones | Proporción |
|---|---:|---:|
| Training | 1,257 | 70% |
| Validation | 270 | 15% |
| Test | 270 | 15% |

Se utilizó `stratify` para conservar aproximadamente la distribución original de las diez clases.

El conjunto de **validation** se utilizó para comparar configuraciones y seleccionar hiperparámetros.

El conjunto de **test** se mantuvo separado del proceso de selección y se utilizó para evaluar la capacidad de generalización del modelo.

---

## Modelo analizado

El algoritmo utilizado fue:

```python
RandomForestClassifier
```

Se mantuvieron constantes los siguientes hiperparámetros:

```python
n_estimators = 100
criterion = "gini"
min_samples_split = 2
min_samples_leaf = 1
max_features = "sqrt"
bootstrap = True
random_state = 42
```

Para analizar el efecto de la complejidad del modelo se modificó:

```python
max_depth
```

Las configuraciones comparadas fueron:

```text
max_depth = 2
max_depth = 5
max_depth = 10
max_depth = None
```

---

## Métrica principal

La métrica principal utilizada fue **Macro F1-score**.

Se eligió porque el problema contiene diez clases y esta métrica permite evaluar el desempeño de cada una de ellas asignándoles el mismo peso.

También se utilizó **Accuracy** como métrica complementaria.

---

## Comparación de complejidad

Los resultados obtenidos fueron:

| Modelo | Train F1 | Validation F1 | Test F1 | Gap Train-Val |
|---|---:|---:|---:|---:|
| `max_depth = 2` | 0.8407 | 0.8089 | 0.8057 | 0.0318 |
| `max_depth = 5` | 0.9721 | 0.9219 | 0.9373 | 0.0502 |
| `max_depth = 10` | 1.0000 | 0.9702 | 0.9626 | 0.0298 |
| `max_depth = None` | 1.0000 | 0.9662 | 0.9664 | 0.0338 |

---

## Diagnóstico de bias, varianza y ajuste

### `max_depth = 2`

- **Bias:** alto
- **Varianza:** baja
- **Nivel de ajuste:** underfitting

El desempeño es relativamente bajo incluso en training, lo que indica que el modelo tiene capacidad insuficiente para representar correctamente el problema.

### `max_depth = 5`

- **Bias:** medio-bajo
- **Varianza:** media
- **Nivel de ajuste:** intermedio

El modelo mejora considerablemente, pero todavía no obtiene el mejor desempeño de validation.

### `max_depth = 10`

- **Bias:** bajo
- **Varianza:** baja-media
- **Nivel de ajuste:** adecuado

Esta configuración obtuvo el mayor Macro F1-score en validation y la menor brecha Train-Validation entre los modelos de mayor capacidad.

### `max_depth = None`

- **Bias:** bajo
- **Varianza:** media
- **Nivel de ajuste:** ligera tendencia al overfitting

El modelo alcanza desempeño perfecto en training, pero aumentar su complejidad no mejora el desempeño en validation.

---

## Regularización

La técnica de regularización utilizada consistió en **limitar la profundidad máxima de los Decision Trees** que forman el Random Forest.

Se comparó:

```text
Antes:
max_depth = None

Después:
max_depth = 10
```

Resultados:

| Métrica | Sin límite | `max_depth = 10` |
|---|---:|---:|
| Train Macro F1 | 1.0000 | 1.0000 |
| Validation Macro F1 | 0.9662 | 0.9702 |
| Gap Train-Val | 0.0338 | 0.0298 |

La regularización produjo:

```text
+0.0040 en Validation Macro F1
-0.0040 en el Gap Train-Validation
```

Esto permitió obtener un modelo menos complejo y con una mejor capacidad de generalización sobre el conjunto de validation.

---

## Modelo final

La configuración seleccionada fue:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    criterion="gini",
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

Después de seleccionar los hiperparámetros con validation, el modelo final fue entrenado utilizando **training + validation**.

La evaluación final sobre el test set produjo:

| Métrica | Resultado |
|---|---:|
| Accuracy | 97.78% |
| Macro Precision | 0.9785 |
| Macro Recall | 0.9776 |
| Macro F1-score | 0.9778 |

El modelo clasificó correctamente:

```text
264 de 270 observaciones
```

---

## Gráficas incluidas

El análisis se respalda mediante gráficas comparativas de:

- Accuracy en training, validation y test
- Macro F1-score en training, validation y test
- Efecto de la regularización mediante `max_depth`
- Matriz de confusión del modelo final

Estas gráficas permiten visualizar el efecto de la complejidad sobre el bias, la varianza y el nivel de ajuste del modelo.

---

## Reporte completo

El análisis completo, incluyendo tablas, gráficas, diagnóstico de bias y varianza, regularización y conclusiones, se encuentra en el siguiente documento:

[**Ver reporte final en PDF**](./Momento de Retroalimentación_ Módulo 2 Análisis y Reporte sobre el desempeño del modelo..pdf)

---

## Archivos principales

```text
/
├── README.md
├── random_forest.py
├── requirements.txt
├── comparacion_accuracy.png
├── comparacion_f1.png
├── regularizacion_random_forest.png
└── Momento de Retroalimentación_ Módulo 2 Análisis y Reporte sobre el desempeño del modelo..pdf
```

---

## Ejecución

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Ejecutar el análisis:

```bash
python random_forest.py
```

El programa genera las métricas de evaluación y las gráficas utilizadas en el reporte.

---

## Conclusión

El análisis mostró que la complejidad del modelo influye directamente en su capacidad de aprendizaje y generalización.

Una profundidad demasiado pequeña produce **underfitting y bias alto**, mientras que permitir árboles sin límite de profundidad incrementa la complejidad sin mejorar el desempeño en validation.

La configuración `max_depth = 10` presentó el mejor equilibrio entre bias y varianza y fue seleccionada como el modelo con mejor nivel de ajuste.

La regularización permitió mejorar el desempeño en validation y reducir la brecha entre training y validation, mientras que el modelo final alcanzó un **Macro F1-score de 0.9778** sobre el test set.

---

## Autor

Aarón Ramírez Pulido
