# crud.py
from sqlalchemy.orm import Session
from models import Usuario, Modelo

def get_user_by_id_or_username(db: Session, identifier: str):
    return db.query(Usuario).filter(
        (Usuario.username == identifier) | (Usuario.identificacion == identifier)
    ).first()

def create_user(db: Session, data: dict):
    user = Usuario(**data)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_or_create_model(db: Session, nombre: str):
    model = db.query(Modelo).filter(Modelo.nombre == nombre).first()
    if not model:
        model = Modelo(nombre=nombre)
        db.add(model)
        db.commit()
        db.refresh(model)
    return model