# models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class Usuario(Base):
    __tablename__ = "usuarios"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    identificacion = Column(String(20), unique=True, nullable=False)

    respuestas = relationship("Respuesta", back_populates="usuario")

class Modelo(Base):
    __tablename__ = "modelos"
    id = Column(Integer, primary_key=True)
    nombre = Column(String(50), unique=True, nullable=False)

class Respuesta(Base):
    __tablename__ = "respuestas"
    id = Column(Integer, primary_key=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"))
    N = Column(Float)
    P = Column(Float)
    K = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    ph = Column(Float)
    rainfall = Column(Float)
    fecha = Column(DateTime, default=datetime.utcnow)

    # ← ¡AHORA AQUÍ ESTÁ EL RESULTADO FINAL! (solo una vez)
    cultivo_final = Column(String(50))
    confianza = Column(Float)
    todos_coinciden = Column(Boolean)

    usuario = relationship("Usuario", back_populates="respuestas")
    predicciones = relationship("Prediccion", back_populates="respuesta")

class Prediccion(Base):
    __tablename__ = "predicciones"
    id = Column(Integer, primary_key=True, index=True)
    respuesta_id = Column(Integer, ForeignKey("respuestas.id"))
    modelo_id = Column(Integer, ForeignKey("modelos.id"))

    # Solo lo individual del modelo
    cultivo_predicho = Column(String(50))
    respuesta = relationship("Respuesta", back_populates="predicciones")
    modelo = relationship("Modelo")