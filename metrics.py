
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Modelos
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
# Importar la herramienta para el análisis de permutación (Método para SVM)
from sklearn.inspection import permutation_importance

# ===================== 1. CARGA Y PREPARACIÓN DE DATOS =====================

# Cargar datos
# Asegúrate de que 'Crop_recommendation.csv' está en el mismo directorio.
try:
    df = pd.read_csv("Crop_recommendation.csv")
except FileNotFoundError:
    print("Error: El archivo 'Crop_recommendation.csv' no fue encontrado.")
    exit()

# Features y target
X = df.drop('label', axis=1)
y = df['label']

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
crop_names = le.classes_  # Nombres de los cultivos para reportes

# Train-test split (Estratificado para balancear las clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Preparar datos escalados SOLO PARA SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===================== 2. DEFINICIÓN Y ENTRENAMIENTO DE MODELOS =====================

# Definir los 3 modelos
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"{name.upper():^60}")
    print(f"{'='*60}")

    # Entrenar el modelo
    if name == "SVM":
        # Usar datos escalados para SVM
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Usar datos originales para Random Forest y XGBoost
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # --- MÉTRICAS DE EVALUACIÓN ---
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=crop_names, output_dict=True)
    print(report)

    # Mostrar métricas
    print(f"Accuracy     : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision    : {report['weighted avg']['precision']:.4f}")
    print(f"Recall       : {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score     : {report['weighted avg']['f1-score']:.4f}")
    print(f"{'-'*60}")

    # Guardar resultados para resumen
    results_df.loc[len(results_df)] = [
        name, 
        acc, 
        report['weighted avg']['precision'], 
        report['weighted avg']['recall'], 
        report['weighted avg']['f1-score']
    ]

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=crop_names, yticklabels=crop_names,
                cbar=False)
    plt.title(f'Matriz de Confusión - {name}', fontsize=16)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ===================== 3. RESUMEN DE IMPORTANCIA DE FEATURES =====================

print("\n" + "="*70)
print("RESUMEN DE RESULTADOS".center(70))
print("="*70)
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("IMPORTANCIA DE FEATURES POR MODELO".center(70))
print("="*70)

feature_names = X.columns.tolist()

for name, model in models.items():
    print(f"\n{'─'*60}")
    print(f"📊 {name}")
    print(f"{'─'*60}")
    
    if name in ["Random Forest", "XGBoost"]:
        # Modelos basados en árboles: Importancia nativa
        importances = model.feature_importances_
        
        # Crear DataFrame y ordenar
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Mostrar en tabla
        print(importance_df.to_string(index=False))
        
        # Gráfico de barras
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        plt.xlabel('Importancia')
        plt.title(f'Importancia de Features - {name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    elif name == "SVM":
        # SVM (kernel RBF): Usar Permutation Importance (método post-hoc)
        print("💡 Calculando Importancia por Permutación para SVM (kernel RBF)...")
        
        # Ejecutar el análisis de permutación
        # Se usan los datos escalados (X_test_scaled) para el modelo SVM
        r = permutation_importance(
            model, 
            X_test_scaled, 
            y_test, 
            n_repeats=10, 
            random_state=42,
            n_jobs=-1
        )
        
        # Crear DataFrame y ordenar
        sorted_idx = r.importances_mean.argsort()[::-1]
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in sorted_idx],
            'Importance (Mean)': r.importances_mean[sorted_idx],
            'Std Dev': r.importances_std[sorted_idx]
        })
        
        print(importance_df.to_string(index=False))
        
        # Gráfico de barras para Permutation Importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance (Mean)'], color='darkred', xerr=importance_df['Std Dev'])
        plt.xlabel('Caída de Accuracy (Importancia)')
        plt.title('Importancia por Permutación - SVM (RBF)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

print("\nAnálisis de entrenamiento y métricas completado.")