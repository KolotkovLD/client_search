import sqlite3

from fastapi import FastAPI, Depends
from pydantic import BaseModel

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uvicorn

import pandas as pd
from time import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

Base = declarative_base()


class ContactsTable(Base):
    __tablename__ = "contacts"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String)


def write_reference_to_db():
    t = time()

    engine = create_engine("sqlite:///contacts.db")
    Base.metadata.create_all(engine)

    session = sessionmaker()
    session.configure(bind=engine)
    s = session()

    try:
        file_name = r"data/contacts.csv"
        data = pd.read_csv(file_name, sep=";", encoding="utf-8")

        for i in data.iterrows():
            record = ContactsTable(**{"id": int(i[1][0]), "name": i[1][1]})
            s.add(record)

        s.commit()
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        s.rollback()  # Rollback the changes on error
    finally:
        s.close()
        elapsed_time = time() - t
        logger.info(f"Time elapsed: {elapsed_time:.2f} s.")


def get_db():
    engine = create_engine("sqlite:///contacts.db")
    with engine.connect() as connection:
        yield connection


class ContactResponse(BaseModel):
    id: int
    name: str
    dist: float


@app.get("/search", response_model=list[ContactResponse])
async def search_contacts(query: str, db: Session = Depends(get_db)):
    df = pd.read_sql_table("contacts", db)

    distances = [levenshtein(query, contact) for contact in df.name]
    min_distance = min(distances)
    closest_contacts = [
        ContactResponse(id=id_, name=name, dist=dist)
        for id_, name, dist in zip(df["id"], df["name"], distances)
        if dist == min_distance
    ]

    logger.info(f"Search query '{query}' returned {len(closest_contacts)} results.")
    return closest_contacts


def levenshtein(s1, s2):
    s1 = str(s1.replace(" ", "").lower())
    s2 = str(s2.replace(" ", "").lower())
    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = [[0 for x in range(cols)] for y in range(rows)]

    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i - 1] == s2[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = (
                    min(distance[i - 1][j], distance[i][j - 1], distance[i - 1][j - 1])
                    + 1
                )

    return distance[rows - 1][cols - 1]


if __name__ == "__main__":
    try:
        logger.info("Starting loading process")
        write_reference_to_db()
    except sqlite3.IntegrityError as ie:
        logger.warning("Data already created")
    uvicorn.run(app, host="0.0.0.0", port=8000)