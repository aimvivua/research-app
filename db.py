import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "app.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Study(Base):
    __tablename__ = 'studies'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())

class Form(Base):
    __tablename__ = 'forms'
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, index=True)
    name = Column(String, index=True)
    schema_json = Column(Text)
    created_at = Column(DateTime, default=func.now())

class Submission(Base):
    __tablename__ = 'submissions'
    id = Column(Integer, primary_key=True, index=True)
    form_id = Column(Integer, index=True)
    data_json = Column(Text)
    created_at = Column(DateTime, default=func.now())

def init_db():
    Base.metadata.create_all(bind=engine)

def get_session():
    return SessionLocal()