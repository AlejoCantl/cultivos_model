from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from schemas import UserRegister, UserLogin, PrediccionRequest, ResultadoFinal, PrediccionOut
from crud import get_user_by_id_or_username, create_user, get_or_create_model
from models import Usuario, Respuesta, Prediccion  # Importamos las clases, no __table__
import joblib
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import List

Base.metadata.create_all(bind=engine)

# Cargar modelos
rf = joblib.load("modelos/random_forest_crop_recommendation.pkl")
xgb = joblib.load("modelos/xgboost_crop_recommendation.pkl")
svm = joblib.load("modelos/svm_crop_recommendation.pkl")
le = joblib.load("modelos/label_encoder_cultivos.pkl")

app = FastAPI(title="🌱 CropAdvisor API - FINAL y FUNCIONANDO")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    
    # Validar que el usuario existe
    user = db.query(Usuario).filter(Usuario.id == user_id).first()
    if not user:
        raise HTTPException(404, "Usuario no encontrado")

    # Preparar datos para el modelo (sin user_id)
    data_dict = request.dict()
    data_dict.pop("user_id")
    df = pd.DataFrame([data_dict])

    # Predicciones
    preds = {
        "Random Forest": le.inverse_transform(rf.predict(df))[0].title(),
        "XGBoost": le.inverse_transform(xgb.predict(df))[0].title(),
        "SVM": le.inverse_transform(svm.predict(df))[0].title(),
    }

    votos = list(preds.values())
    ganador, conteo = Counter(votos).most_common(1)[0]
    confianza = round(conteo / 3.0, 2)
    todos_coinciden = confianza == 1.0

# Guardar en BD (AHORA TODO PERFECTO)
    respuesta = Respuesta(
        usuario_id=user_id,
        **data_dict,
        cultivo_final=ganador,
        confianza=confianza,
        todos_coinciden=todos_coinciden
    )
    db.add(respuesta)
    db.commit()
    db.refresh(respuesta)

    # Guardar solo las predicciones individuales
    for nombre_modelo, cultivo_individual in preds.items():
        modelo_db = get_or_create_model(db, nombre_modelo)
        pred = Prediccion(
            respuesta_id=respuesta.id,
            modelo_id=modelo_db.id,
            cultivo_predicho=cultivo_individual,
        )
        db.add(pred)
    db.commit()

    return ResultadoFinal(
        cultivo_recomendado=ganador,
        confianza=confianza,
        todos_coinciden=todos_coinciden,
        predicciones=[PrediccionOut(modelo=k, cultivo=v) for k, v in preds.items()],
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
            "cultivo_final": preds[0].cultivo_final if preds else None,
            "confianza": preds[0].confianza if preds else None,
            "todos_coinciden": preds[0].todos_coinciden if preds else None,
            "predicciones": [{"modelo": p.modelo, "cultivo": p.cultivo_predicho} for p in preds]
        })
    return resultados