import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Modelos
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Cargar datos
df = pd.read_csv("Crop_recommendation.csv")

# ===================== VALIDACIÓN DE DATOS =====================
print("INFORMACIÓN GENERAL DEL DATASET")
print("="*60)
print(df.info())
print("\n")

print("DIMENSIÓN DEL DATASET")
print("="*60)
print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}\n")

print("VALORES NULOS POR COLUMNA")
print("="*60)
print(df.isnull().sum())
print("\n")

print("¿HAY ALGÚN VALOR NULO EN TODO EL DATASET?")
print("="*60)
print(f"Total de valores nulos: {df.isnull().sum().sum()}")
print(f"¿Existe algún NaN? → {'SÍ ❌' if df.isnull().any().any() else 'NO ✅'}\n")

print("FILAS DUPLICADAS")
print("="*60)
duplicados = df.duplicated().sum()
print(f"Número de filas completamente duplicadas: {duplicados}")
if duplicados > 0:
    print("Mostrando las filas duplicadas:")
    print(df[df.duplicated(keep=False)])  # muestra todas las copias
else:
    print("¡No hay duplicados! ✅\n")

print("ESTADÍSTICAS BÁSICAS (para detectar outliers o valores raros)")
print("="*60)
print(df.describe())

print("\nTIPO DE DATOS POR COLUMNA")
print("="*60)
print(df.dtypes)

print("\nNÚMERO DE CLASES (cultivos) Y DISTRIBUCIÓN")
print("="*60)
print(f"Total de cultivos únicos: {df['label'].nunique()}")
print(df['label'].value_counts())
print("\n¡El dataset está 100% limpio y balanceado!" if (df.isnull().sum().sum() == 0 and duplicados == 0) else "Hay que limpiar algo...")

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

from sklearn.preprocessing import StandardScaler

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

# Entrenar y evaluar cada modelo
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"{name.upper():^60}")
    print(f"{'='*60}")

# ✨ ÚNICO CAMBIO: usar datos escalados solo si es SVM
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy     : {acc:.4f} ({acc*100:.2f}%)")

    # Classification report completo
    report = classification_report(y_test, y_pred, target_names=crop_names, output_dict=True)
    print(f"Precision    : {report['weighted avg']['precision']:.4f}")
    print(f"Recall       : {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score     : {report['weighted avg']['f1-score']:.4f}")

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

    # ===================== IMPORTANCIA DE FEATURES =====================
print("\n" + "="*60)
print("IMPORTANCIA DE FEATURES POR MODELO".center(60))
print("="*60)

feature_names = X.columns.tolist()

for name, model in models.items():
    print(f"\n{'─'*60}")
    print(f"📊 {name}")
    print(f"{'─'*60}")
    
    if name == "Random Forest":
        importances = model.feature_importances_
        
    elif name == "XGBoost":
        importances = model.feature_importances_
        
    elif name == "SVM":
        # SVM con kernel RBF no tiene feature importance directa
        # Pero podemos usar los coeficientes del dual (aproximación)
        print("⚠️  SVM (kernel RBF) no tiene importancia de features directa")
        print("    (usa transformaciones no lineales en espacio de alta dimensión)")
        continue
    
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

print("\n" + "="*60)

import joblib
import os

# Crear carpeta si no existe
os.makedirs("modelos", exist_ok=True)

# ================== GUARDAR RANDOM FOREST (el más estable y rápido) ==================
joblib.dump(models["Random Forest"], "modelos/random_forest_crop_recommendation.pkl")
print("✅ Random Forest guardado → modelos/random_forest_crop_recommendation.pkl")

# ================== GUARDAR XGBOOST (a veces da ligeramente más accuracy) ==================
joblib.dump(models["XGBoost"], "modelos/xgboost_crop_recommendation.pkl")
print("✅ XGBoost guardado → modelos/xgboost_crop_recommendation.pkl")

# ================== SVM ================================================
joblib.dump(models["SVM"], "modelos/svm_crop_recommendation.pkl")
print("✅ SVM guardado → modelos/svm_crop_recommendation.pkl")

# ================== GUARDAR EL SCALER (necesario para usar SVM después) ==================
joblib.dump(scaler, "modelos/scaler_svm.pkl")
print("✅ Scaler guardado → modelos/scaler_svm.pkl")

# ================== GUARDAR EL LABEL ENCODER (imprescindible!) ==================
joblib.dump(le, "modelos/label_encoder_cultivos.pkl")
print("✅ LabelEncoder guardado → modelos/label_encoder_cultivos.pkl")

print("\n¡Todo guardado correctamente!")
print("Ahora puedes usar los modelos en cualquier momento con solo cargarlos.")

# Cargar todo al inicio
rf = joblib.load("modelos/random_forest_crop_recommendation.pkl")
xgb = joblib.load("modelos/xgboost_crop_recommendation.pkl")
svm = joblib.load("modelos/svm_crop_recommendation.pkl")
scaler = joblib.load("modelos/scaler_svm.pkl")  # ✨ NUEVO
le = joblib.load("modelos/label_encoder_cultivos.pkl")

def recomendar_cultivo(N, P, K, temperature, humidity, ph, rainfall, modelo="svm"):
    datos = pd.DataFrame([{
        'N': N, 'P': P, 'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])
    
    # Elegir modelo
    if modelo.lower() == "random forest":
        pred = rf.predict(datos)[0]
    elif modelo.lower() == "xgboost":
        pred = xgb.predict(datos)[0]
    elif modelo.lower() == "svm":
        datos_scaled = scaler.transform(datos)  # ✨ Escalar solo para SVM
        pred = svm.predict(datos_scaled)[0]
    
    cultivo = le.inverse_transform([pred])[0]
    return cultivo.upper()

ejemplos = [
    # 1. Clásico para arroz (debería dar RICE)
    (90, 42, 43, 20.8, 82, 6.5, 202),

    # 2. Ideal para maíz (MAIZE)
    (70, 50, 20, 23, 65, 6.5, 90),

    # 3. Perfecto para café (COFFEE)
    (100, 25, 30, 25, 60, 6.8, 150),

    # 4. Algodón en clima cálido (COTTON)
    (120, 40, 20, 24, 80, 7.0, 85),

    # 5. Condiciones para judías/verdes (KIDNEYBEANS)
    (20, 65, 20, 20, 70, 6.0, 110),

    # 6. Garbanzos en clima seco (CHICKPEA)
    (40, 60, 80, 18, 16, 7.2, 80),

    # 7. Yute en zona muy húmeda (JUTE)
    (80, 45, 40, 25, 85, 6.8, 180),

    # 8. Coco tropical (COCONUT)
    (25, 15, 30, 27, 95, 6.0, 200),

    # 9. Manzana (APPLE) - clima templado/frío
    (20, 135, 200, 22, 90, 6.0, 120),  # K muy alto = típico de frutales

    # 10. Uvas (GRAPES) - clima mediterráneo
    (65, 120, 200, 23, 75, 6.2, 95),
]

print("🌱 PREDICCIONES CON TU MODELO 🌱\n" + "="*60)
for i, valores in enumerate(ejemplos, 1):
    N, P, K, temp, hum, ph, rain = valores
    cultivo = recomendar_cultivo(N, P, K, temp, hum, ph, rain)
    print(f"{i:2}. N={N:3} P={P:3} K={K:3} | Temp={temp}°C Hum={hum}% pH={ph} Rain={rain}mm → {cultivo}")