# schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import List

class UserRegister(BaseModel):
    username: str
    password: str
    nombre: str
    apellido: str
    identificacion: str

class UserLogin(BaseModel):
    identificacion_o_username: str
    password: str

class SoilConditions(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class PrediccionOut(BaseModel):
    modelo: str
    cultivo: str

class ResultadoFinal(BaseModel):
    cultivo_recomendado: str
    confianza: float
    todos_coinciden: bool
    predicciones: List[PrediccionOut]
    fecha: datetime

class PrediccionRequest(BaseModel):
    user_id: int
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float