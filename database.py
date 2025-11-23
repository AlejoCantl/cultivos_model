# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Cambia estos valores según tu MySQL
import os
from dotenv import load_dotenv
load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "recomendacion_cultivo")

DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# Configuración optimizada para conexiones limitadas
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,          # Verifica que la conexión esté viva antes de usarla
    pool_recycle=3600,            # Recicla conexiones cada hora
    pool_size=3,                  # Reduce a 3 conexiones en el pool
    max_overflow=0,               # Sin conexiones extras
    pool_timeout=30,              # Timeout de 30 segundos esperando conexión
    echo=False,                   # Desactiva logging SQL para mejor performance
    connect_args={
        'connect_timeout': 10,    # Timeout de conexión de 10 segundos
        'autocommit': False,
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()