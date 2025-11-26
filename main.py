from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from schemas import UserRegister, UserLogin, PrediccionRequest, ResultadoFinal, PrediccionOut
from crud import get_user_by_id_or_username, create_user, get_or_create_model
from models import Usuario, Respuesta, Prediccion, Modelo
import joblib
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import List
import os

Base.metadata.create_all(bind=engine)

# Cargar modelos
rf = joblib.load("modelos/random_forest_crop_recommendation.pkl")
xgb = joblib.load("modelos/xgboost_crop_recommendation.pkl")
svm = joblib.load("modelos/svm_crop_recommendation.pkl")
scaler = joblib.load("modelos/scaler_svm.pkl")
le = joblib.load("modelos/label_encoder_cultivos.pkl")

app = FastAPI(title="🌱 CropAdvisor API - FINAL y FUNCIONANDO")

# ====================== CONFIGURACIÓN DE CORS ======================
# Configuración para desarrollo y producción
origins = [
    "*"  # Permitir todos los orígenes (solo para desarrollo/testing)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Lista de orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

def get_db():
    """Generador de sesiones de base de datos con cierre garantizado"""
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Commit explícito si todo salió bien
    except Exception:
        db.rollback()  # Rollback en caso de error
        raise
    finally:
        db.close()  # Siempre cierra la conexión

# ====================== ENDPOINT DE SALUD ======================
@app.get("/")
def root():
    return {
        "message": "🌱 CropAdvisor API está funcionando",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.head("/health")
def health_check_head():
    return {"status": "ok"}

# ====================== REGISTRO Y LOGIN ======================
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserRegister, db: Session = Depends(get_db)):
    if get_user_by_id_or_username(db, user.username):
        raise HTTPException(400, "Username ya existe")
    if get_user_by_id_or_username(db, user.identificacion):
        raise HTTPException(400, "Identificación ya registrada")
    create_user(db, user.dict())
    return {"message": "Usuario creado correctamente"}

@app.post("/login")
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = get_user_by_id_or_username(db, credentials.identificacion_o_username)
    if not user or user.password != credentials.password:
        raise HTTPException(401, "Credenciales incorrectas")
    return {"user_id": user.id, "nombre": user.nombre, "apellido": user.apellido, "message": "Login exitoso"}

# ====================== PREDICCIÓN (AHORA SÍ FUNCIONA) ======================
@app.post("/predecir", response_model=ResultadoFinal)
def predecir(request: PrediccionRequest, db: Session = Depends(get_db)):
    user_id = request.user_id
    
    # Validar usuario
    user = db.query(Usuario).filter(Usuario.id == user_id).first()
    if not user:
        raise HTTPException(404, "Usuario no encontrado")

    # Preparar datos para los modelos
    data_dict = request.dict()
    data_dict.pop("user_id")
    df = pd.DataFrame([data_dict])

    # Escalado para SVM
    df_array = df.values
    df_scaled = scaler.transform(df_array)

    # Predicciones en inglés
    preds_en = {
        "Random Forest": le.inverse_transform(rf.predict(df))[0].title(),
        "XGBoost":       le.inverse_transform(xgb.predict(df))[0].title(),
        "SVM":           le.inverse_transform(svm.predict(df_scaled))[0].title(),
    }

    # Votación
    votos = list(preds_en.values())
    ganador_en, conteo = Counter(votos).most_common(1)[0]
    confianza = round(conteo / 3.0, 2)
    todos_coinciden = confianza == 1.0

    # === TRADUCCIÓN AL ESPAÑOL ===
    traduccion_cultivos = {
        "Rice": "Arroz", "Maize": "Maíz", "Chickpea": "Garbanzo", "Kidneybeans": "Frijol rojo",
        "Pigeonpeas": "Gandul", "Mothbeans": "Frijol moth", "Mungbean": "Frijol mungo",
        "Blackgram": "Frijol negro", "Lentil": "Lenteja", "Pomegranate": "Granada",
        "Banana": "Plátano", "Mango": "Mango", "Grapes": "Uva", "Watermelon": "Sandía",
        "Muskmelon": "Melón", "Apple": "Manzana", "Orange": "Naranja", "Papaya": "Papaya",
        "Coconut": "Coco", "Cotton": "Algodón", "Jute": "Yute", "Coffee": "Café"
    }

    ganador_es = traduccion_cultivos.get(ganador_en, ganador_en)
    preds_es = {modelo: traduccion_cultivos.get(cultivo, cultivo) for modelo, cultivo in preds_en.items()}

    # Guardar en BD (cultivo final en español)
    respuesta = Respuesta(
        usuario_id=user_id,
        **data_dict,
        cultivo_final=ganador_es,
        confianza=confianza,
        todos_coinciden=todos_coinciden
    )
    db.add(respuesta)
    db.commit()
    db.refresh(respuesta)

    # Guardar predicciones individuales (en español también)
    for nombre_modelo, cultivo_en in preds_en.items():
        modelo_db = get_or_create_model(db, nombre_modelo)
        pred = Prediccion(
            respuesta_id=respuesta.id,
            modelo_id=modelo_db.id,
            cultivo_predicho=traduccion_cultivos.get(cultivo_en, cultivo_en),
        )
        db.add(pred)
    db.commit()

    # Respuesta al usuario (todo en español)
    return ResultadoFinal(
        cultivo_recomendado=ganador_es,
        confianza=confianza,
        todos_coinciden=todos_coinciden,
        predicciones=[PrediccionOut(modelo=modelo, cultivo=cultivo_es) for modelo, cultivo_es in preds_es.items()],
        fecha=datetime.utcnow()
    )

# ====================== HISTORIAL ======================
@app.get("/historial/{user_id}")
def historial(user_id: int, db: Session = Depends(get_db)):
    respuestas = db.query(Respuesta).filter(Respuesta.usuario_id == user_id).order_by(Respuesta.fecha.desc()).all()
    resultados = []
    for r in respuestas:
        preds = db.query(Prediccion).filter(Prediccion.respuesta_id == r.id).all()
        resultados.append({
            "respuesta_id": r.id,
            "fecha": r.fecha,
            "input": {"N": r.N, "P": r.P, "K": r.K, "temperature": r.temperature,
                      "humidity": r.humidity, "ph": r.ph, "rainfall": r.rainfall},
            "cultivo_final": r.cultivo_final,
            "confianza": r.confianza,
            "todos_coinciden": r.todos_coinciden,
            "predicciones": [{"modelo": p.modelo, "cultivo": p.cultivo_predicho} for p in preds]
        })
    return resultados

# ====================== ENVÍO DE CORREO ======================
from email_utils import generate_history_pdf, send_history_email
from schemas import EmailRequest

@app.post("/send-history")
def send_history(request: EmailRequest, db: Session = Depends(get_db)):
    # 1. Validar usuario
    user = db.query(Usuario).filter(Usuario.id == request.user_id).first()
    if not user:
        raise HTTPException(404, "Usuario no encontrado")

    # 2. Obtener historial
    query = db.query(Respuesta).filter(Respuesta.usuario_id == request.user_id).order_by(Respuesta.fecha.desc())
    respuestas = query.all()
    
    if not respuestas:
        raise HTTPException(404, "No hay historial para este usuario")

    history_data = []
    for r in respuestas:
        preds_query = db.query(Prediccion).filter(Prediccion.respuesta_id == r.id)
        
        # Filtrar por modelo si se especifica
        if request.model_filter:
            preds_query = preds_query.join(Modelo).filter(Modelo.nombre == request.model_filter)
        
        preds = preds_query.all()
        
        # Si hay filtro y esta respuesta no tiene predicción de ese modelo, saltarla
        if request.model_filter and not preds:
            continue

        history_data.append({
            "fecha": r.fecha,
            "cultivo_final": r.cultivo_final,
            "confianza": r.confianza,
            "input": {"N": r.N, "P": r.P, "K": r.K, "temperature": r.temperature,
                      "humidity": r.humidity, "ph": r.ph, "rainfall": r.rainfall},
            "predicciones": [{"modelo": p.modelo.nombre, "cultivo": p.cultivo_predicho} for p in preds]
        })

    if not history_data:
        mensaje = f"No se encontró historial para el modelo '{request.model_filter}'" if request.model_filter else "No hay historial para este usuario"
        raise HTTPException(404, mensaje)

    # 3. Generar PDF (con filtro de modelo si aplica)
    filename = f"historial_{request.user_id}_{request.model_filter or 'completo'}.pdf"
    try:
        pdf_path = generate_history_pdf(history_data, filename, model_filter=request.model_filter)
    except Exception as e:
        raise HTTPException(500, f"Error generando PDF: {str(e)}")

    # 4. Enviar Correo (con filtro de modelo si aplica)
    try:
        user_full_name = f"{user.nombre} {user.apellido}"
        send_history_email(request.email, pdf_path, user_full_name, model_filter=request.model_filter)
    except Exception as e:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(500, f"Error enviando correo: {str(e)}")

    # 5. Limpiar archivo
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    mensaje_respuesta = f"Historial del modelo '{request.model_filter}' enviado a {request.email}" if request.model_filter else f"Historial completo enviado a {request.email}"
    return {"message": mensaje_respuesta}



# ====================== CONFIGURACIÓN PARA RENDER ======================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)