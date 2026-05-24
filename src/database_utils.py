import sqlite3
from typing import List


def create_database(file_path: str):
    """
    Creates a SQLite database with a table for storing text chunks and their embeddings.
    The table has the following columns:
    - id: an auto-incrementing primary key
    - chunk: the text chunk (string)
    - embedding: the embedding vector (stored as a JSON string)
    """

    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def insert_chunk(file_path: str, chunk: str, embedding: str):
    """
    Inserts a text chunk and its embedding into the database.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO chunks (chunk, embedding) VALUES (?, ?)
    ''', (chunk, embedding))

    conn.commit()
    conn.close()

def insert_chunks(file_path: str, chunks_list: List[str], embedding: str):
    """
    Inserts a text chunk and its embedding into the database.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    for chunk in chunks_list:
        cursor.execute('''
            INSERT INTO chunks (chunk, embedding) VALUES (?, ?)
        ''', (chunk, embedding))

    conn.commit()
    conn.close()

def get_all_chunks(file_path: str):
    """
    Retrieves all text chunks and their embeddings from the database.
    Returns a list of tuples (chunk, embedding).
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('SELECT chunk, embedding FROM chunks')
    rows = cursor.fetchall()

    conn.close()
    return rows