"""
Clasificación de dígitos con Random Forest usando Scikit-learn.

Dataset: Digits Dataset de Scikit-learn.

El objetivo es clasificar imágenes de dígitos escritos a mano
en una de diez clases posibles: 0, 1, 2, ..., 9.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def contar_clases(labels):
    """
    Cuenta cuántas observaciones pertenecen a cada clase.
    """

    conteo = {}

    for label in labels:
        conteo[int(label)] = conteo.get(int(label), 0) + 1

    return conteo

def seleccionar_modelo(
    x_train,
    y_train,
    x_validation,
    y_validation
):
    """
    Prueba diferentes configuraciones de Random Forest
    y selecciona la que obtiene el mayor Macro F1-score
    sobre el validation set.
    """

    numeros_arboles = [50, 100, 200]
    profundidades = [5, 10, None]

    mejor_modelo = None
    mejor_configuracion = None
    mejor_f1 = -1.0

    resultados = []

    for n_estimators in numeros_arboles:

        for max_depth in profundidades:

            modelo = RandomForestClassifier(
                # Número de árboles que forman el Random Forest
                n_estimators=n_estimators,

                # Criterio utilizado para evaluar las divisiones
                criterion="gini",

                # Limita la profundidad máxima de cada árbol
                max_depth=max_depth,

                # Mínimo de observaciones necesarias para dividir un nodo
                min_samples_split=2,

                # Mínimo de observaciones permitidas en cada nodo hoja
                min_samples_leaf=1,

                # Número de features considerados en cada división
                max_features="sqrt",

                # Cada árbol se entrena con una muestra bootstrap
                bootstrap=True,

                # Permite reproducir los mismos resultados
                random_state=42,

                # Utiliza todos los núcleos disponibles del procesador
                n_jobs=-1
            )

            # Entrenar el modelo
            modelo.fit(
                x_train,
                y_train
            )

            # Predicciones sobre training
            pred_train = modelo.predict(
                x_train
            )

            # Predicciones sobre validation
            pred_validation = modelo.predict(
                x_validation
            )

            # Accuracy en ambos conjuntos
            train_accuracy = accuracy_score(
                y_train,
                pred_train
            )

            validation_accuracy = accuracy_score(
                y_validation,
                pred_validation
            )

            train_f1 = f1_score(
                y_train,
                pred_train,
                average="macro"
            )

            validation_f1 = f1_score(
                y_validation,
                pred_validation,
                average="macro"
            )

            resultados.append(
                (
                    n_estimators,
                    max_depth,
                    train_accuracy,
                    validation_accuracy,
                    train_f1,
                    validation_f1
                )
            )

            # Seleccionar usando únicamente validation
            if validation_f1 > mejor_f1:
                mejor_f1 = validation_f1
                mejor_modelo = modelo

                mejor_configuracion = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

    return (
        mejor_modelo,
        mejor_configuracion,
        mejor_f1,
        resultados
    )

def analizar_niveles_ajuste(
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test
):
    """
    Compara tres niveles de complejidad del Random Forest
    para analizar bias, varianza y nivel de ajuste.

    Se mantiene fijo n_estimators=100 y se modifica
    únicamente max_depth.
    """

    configuraciones = [
        ("Profundidad 2", 2),
        ("Profundidad 5", 5),
        ("Profundidad 10", 10),
        ("Sin límite", None)
    ]

    resultados = []

    for nombre, profundidad in configuraciones:

        modelo = RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=profundidad,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        # Entrenar únicamente con el training set
        modelo.fit(
            x_train,
            y_train
        )

        # Predicciones en los tres conjuntos
        pred_train = modelo.predict(x_train)
        pred_validation = modelo.predict(x_validation)
        pred_test = modelo.predict(x_test)

        # Accuracy
        train_accuracy = accuracy_score(
            y_train,
            pred_train
        )

        validation_accuracy = accuracy_score(
            y_validation,
            pred_validation
        )

        test_accuracy = accuracy_score(
            y_test,
            pred_test
        )

        # Macro F1-score
        train_f1 = f1_score(
            y_train,
            pred_train,
            average="macro"
        )

        validation_f1 = f1_score(
            y_validation,
            pred_validation,
            average="macro"
        )

        test_f1 = f1_score(
            y_test,
            pred_test,
            average="macro"
        )

        # Diferencia entre training y validation.
        # Una brecha mayor puede indicar mayor varianza.
        brecha_f1 = train_f1 - validation_f1

        resultados.append({
            "nombre": nombre,
            "max_depth": profundidad,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "train_f1": train_f1,
            "validation_f1": validation_f1,
            "test_f1": test_f1,
            "brecha_f1": brecha_f1
        })

    return resultados

def generar_graficas_ajuste(resultados):
    """
    Genera gráficas comparativas para analizar bias,
    varianza, nivel de ajuste y efecto de la regularización.
    """

    nombres = [
        resultado["nombre"]
        for resultado in resultados
    ]

    train_f1 = [
        resultado["train_f1"]
        for resultado in resultados
    ]

    validation_f1 = [
        resultado["validation_f1"]
        for resultado in resultados
    ]

    test_f1 = [
        resultado["test_f1"]
        for resultado in resultados
    ]

    train_accuracy = [
        resultado["train_accuracy"]
        for resultado in resultados
    ]

    validation_accuracy = [
        resultado["validation_accuracy"]
        for resultado in resultados
    ]

    test_accuracy = [
        resultado["test_accuracy"]
        for resultado in resultados
    ]

    # ----------------------------------------------------------
    # Gráfica 1: Macro F1 según la complejidad
    # ----------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        nombres,
        train_f1,
        marker="o",
        label="Training"
    )

    plt.plot(
        nombres,
        validation_f1,
        marker="o",
        label="Validation"
    )

    plt.plot(
        nombres,
        test_f1,
        marker="o",
        label="Test"
    )

    plt.title(
        "Macro F1 según la complejidad del Random Forest"
    )

    plt.xlabel("Configuración de max_depth")
    plt.ylabel("Macro F1-score")
    plt.ylim(0.75, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "comparacion_f1.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------------------
    # Gráfica 2: Accuracy según la complejidad
    # ----------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        nombres,
        train_accuracy,
        marker="o",
        label="Training"
    )

    plt.plot(
        nombres,
        validation_accuracy,
        marker="o",
        label="Validation"
    )

    plt.plot(
        nombres,
        test_accuracy,
        marker="o",
        label="Test"
    )

    plt.title(
        "Accuracy según la complejidad del Random Forest"
    )

    plt.xlabel("Configuración de max_depth")
    plt.ylabel("Accuracy")
    plt.ylim(0.75, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "comparacion_accuracy.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------------------
    # Gráfica 3: Efecto de la regularización
    # ----------------------------------------------------------

    modelo_regularizado = next(
        resultado
        for resultado in resultados
        if resultado["max_depth"] == 10
    )

    modelo_sin_limite = next(
        resultado
        for resultado in resultados
        if resultado["max_depth"] is None
    )

    modelos = [
        "Sin límite",
        "max_depth=10"
    ]

    train_regularizacion = [
        modelo_sin_limite["train_f1"],
        modelo_regularizado["train_f1"]
    ]

    validation_regularizacion = [
        modelo_sin_limite["validation_f1"],
        modelo_regularizado["validation_f1"]
    ]

    posiciones = range(len(modelos))
    ancho = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        [x - ancho / 2 for x in posiciones],
        train_regularizacion,
        width=ancho,
        label="Training"
    )

    plt.bar(
        [x + ancho / 2 for x in posiciones],
        validation_regularizacion,
        width=ancho,
        label="Validation"
    )

    plt.xticks(
        list(posiciones),
        modelos
    )

    plt.ylim(0.90, 1.01)

    plt.ylabel("Macro F1-score")

    plt.title(
        "Efecto de limitar max_depth"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "regularizacion_random_forest.png",
        dpi=300
    )

    plt.close()

    print("\nGráficas generadas correctamente:")
    print("- comparacion_f1.png")
    print("- comparacion_accuracy.png")
    print("- regularizacion_random_forest.png")

def main():
    """
    Carga el Digits Dataset y lo divide en conjuntos
    de training, validation y test.
    """

    # ----------------------------------------------------------
    # 1. Cargar el Digits Dataset
    # ----------------------------------------------------------

    digits = load_digits()

    features = digits.data
    labels = digits.target

    print("Scikit-learn Digits Dataset")
    print("---------------------------")
    print(f"Número de observaciones: {len(features)}")
    print(f"Número de features: {features.shape[1]}")
    clases = [int(clase) for clase in sorted(set(labels))]
    print(f"Clases encontradas: {clases}")

    # ----------------------------------------------------------
    # 2. Separar training del resto de los datos
    # ----------------------------------------------------------

    (
        x_train,
        x_temp,
        y_train,
        y_temp
    ) = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    # ----------------------------------------------------------
    # 3. Dividir el 30% restante en validation y test
    # ----------------------------------------------------------

    (
        x_validation,
        x_test,
        y_validation,
        y_test
    ) = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nDivisión del dataset")
    print("--------------------")
    print(f"Training:   {len(x_train)} observaciones")
    print(f"Validation: {len(x_validation)} observaciones")
    print(f"Test:       {len(x_test)} observaciones")

    print("\nDistribución de clases")
    print("----------------------")
    print(f"Training:   {contar_clases(y_train)}")
    print(f"Validation: {contar_clases(y_validation)}")
    print(f"Test:       {contar_clases(y_test)}")

    # ----------------------------------------------------------
    # 4. Selección de hiperparámetros
    # ----------------------------------------------------------

    print("\nSelección de hiperparámetros")
    print("----------------------------")

    (
        mejor_modelo,
        mejor_configuracion,
        mejor_f1,
        resultados
    ) = seleccionar_modelo(
        x_train,
        y_train,
        x_validation,
        y_validation
    )

    for (
        n_estimators,
        max_depth,
        train_accuracy,
        validation_accuracy,
        train_f1,
        validation_f1
    ) in resultados:

        print(
            f"Árboles: {n_estimators:3d} | "
            f"Profundidad: {str(max_depth):>4} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | "
            f"Val F1: {validation_f1:.4f}"
        )

    print("\nMejor configuración")
    print("-------------------")

    print(
        f"n_estimators: "
        f"{mejor_configuracion['n_estimators']}"
    )

    print(
        f"max_depth: "
        f"{mejor_configuracion['max_depth']}"
    )

    print(
        f"Validation Macro F1: "
        f"{mejor_f1:.4f}"
    )

    # ----------------------------------------------------------
    # 5. Entrenamiento del modelo final
    # ----------------------------------------------------------

    print("\nEntrenamiento del modelo final")
    print("------------------------------")

    modelo_final = RandomForestClassifier(
        n_estimators=mejor_configuracion["n_estimators"],
        criterion="gini",
        max_depth=mejor_configuracion["max_depth"],
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # Utilizar training + validation para entrenar
    # el modelo final con más información disponible.
    # Después de seleccionar los hiperparámetros con validation,
    # se combinan training y validation para entrenar el modelo final.
    # El test set permanece separado y solo se utiliza para
    # la evaluación final del desempeño.
    x_final_train = list(x_train) + list(x_validation)
    y_final_train = list(y_train) + list(y_validation)

    modelo_final.fit(
        x_final_train,
        y_final_train
    )

    print(
        f"Observaciones usadas para entrenamiento final: "
        f"{len(x_final_train)}"
    )

    print("Modelo final entrenado correctamente.")

    # ----------------------------------------------------------
    # 6. Predicciones sobre el test set
    # ----------------------------------------------------------

    predicciones_test = modelo_final.predict(
        x_test
    )

    print("\nPrimeras 10 predicciones")
    print("------------------------")

    for i in range(10):
        print(
            f"Ejemplo {i + 1}: "
            f"Real = {int(y_test[i])} | "
            f"Predicción = {int(predicciones_test[i])}"
        )

    # ----------------------------------------------------------
    # 7. Matriz de confusión
    # ----------------------------------------------------------

    matriz = confusion_matrix(
        y_test,
        predicciones_test
    )

    print("\nMatriz de confusión")
    print("-------------------")
    print(matriz)

    # ----------------------------------------------------------
    # 8. Métricas de evaluación
    # ----------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predicciones_test
    )

    macro_precision = precision_score(
        y_test,
        predicciones_test,
        average="macro"
    )

    macro_recall = recall_score(
        y_test,
        predicciones_test,
        average="macro"
    )

    # Macro F1-score se utiliza como métrica principal porque
    # el problema contiene 10 clases y se busca dar el mismo
    # peso al desempeño obtenido en cada una de ellas.

    macro_f1 = f1_score(
        y_test,
        predicciones_test,
        average="macro"
    )

    print("\nMétricas finales")
    print("----------------")

    print(
        f"Accuracy:        "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1-score:  "
        f"{macro_f1:.4f}"
    )

    print("\nReporte por clase")
    print("-----------------")

    print(
        classification_report(
            y_test,
            predicciones_test,
            digits=4
        )
    )

    # ----------------------------------------------------------
    # 9. Análisis de bias, varianza y nivel de ajuste
    # ----------------------------------------------------------

    print("\nAnálisis comparativo de nivel de ajuste")
    print("---------------------------------------")

    resultados_ajuste = analizar_niveles_ajuste(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test
    )

    print(
        f"{'Modelo':<18}"
        f"{'Train Acc':>12}"
        f"{'Val Acc':>12}"
        f"{'Test Acc':>12}"
        f"{'Train F1':>12}"
        f"{'Val F1':>12}"
        f"{'Test F1':>12}"
        f"{'Gap F1':>10}"
    )

    print("-" * 100)

    for resultado in resultados_ajuste:

        print(
            f"{resultado['nombre']:<18}"
            f"{resultado['train_accuracy']:>12.4f}"
            f"{resultado['validation_accuracy']:>12.4f}"
            f"{resultado['test_accuracy']:>12.4f}"
            f"{resultado['train_f1']:>12.4f}"
            f"{resultado['validation_f1']:>12.4f}"
            f"{resultado['test_f1']:>12.4f}"
            f"{resultado['brecha_f1']:>10.4f}"
        )

    # ----------------------------------------------------------
    # 10. Generación de gráficas comparativas
    # ----------------------------------------------------------

    generar_graficas_ajuste(
        resultados_ajuste
    )

if __name__ == "__main__":
    main()
