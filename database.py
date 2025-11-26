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

# database.py

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True, 
    pool_recycle=3600,
    # === REDUCCIÓN ADICIONAL PARA CUMPLIR CON EL LÍMITE DE 5 ===
    pool_size=2,          # ¡Reducir de 3 a 2 conexiones máximas en el pool!
    max_overflow=0,       # Mantener en 0
    pool_timeout=15,      # Reducir el timeout a 15s
    # ==========================================================
    echo=False,
    connect_args={
        'connect_timeout': 10,
        'autocommit': False,
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()