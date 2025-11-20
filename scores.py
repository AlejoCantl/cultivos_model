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

# Cargar datos
df = pd.read_csv("Crop_recommendation.csv")

# ===================== VALIDACIÓN DE DATOS (Omisión de prints para brevedad) =====================
# El código de validación original confirma que el dataset está limpio y balanceado.

# Features y target
X = df.drop('label', axis=1)
y = df['label']

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
crop_names = le.classes_  # Para mostrar nombres reales en la matriz

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# ===================== PREPARAR DATOS ESCALADOS SOLO PARA SVM =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Definir los 3 modelos
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

# Diccionario para almacenar las métricas de cada modelo
metrics_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Entrenar y evaluar cada modelo
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"{name.upper():^60}")
    print(f"{'='*60}")

    # Usar datos escalados solo si es SVM
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=crop_names, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    # Imprimir métricas
    print(f"Accuracy     : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-Score     : {f1_score:.4f}")

    # Almacenar métricas para el gráfico comparativo
    metrics_data['Model'].append(name)
    metrics_data['Accuracy'].append(acc)
    metrics_data['Precision'].append(precision)
    metrics_data['Recall'].append(recall)
    metrics_data['F1-Score'].append(f1_score)

    # Matriz de confusión con nombres de cultivos
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

    print(f"{'-'*60}")

# ===================== GRÁFICO COMPARATIVO DE MÉTRICAS =====================

metrics_df = pd.DataFrame(metrics_data).set_index('Model')

print("\n" + "="*60)
print("COMPARATIVA DE MÉTRICAS POR MODELO".center(60))
print("="*60)
print(metrics_df.round(4)) # Mostrar la tabla de métricas

# Crear el gráfico de barras comparativo
metrics_df.plot(kind='bar', figsize=(14, 8), colormap='viridis')
plt.title('Comparativa de Métricas de Rendimiento por Modelo', fontsize=18)
plt.ylabel('Puntuación de Métrica', fontsize=12)
plt.xlabel('Modelo', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()