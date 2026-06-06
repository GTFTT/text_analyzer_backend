import json
import sqlite3


def create_database(file_path: str):
    """
    Creates a SQLite database with a table for storing text chunks and their embeddings.
    """

    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk TEXT NOT NULL,
            embedding TEXT NOT NULL,
            project_id INTEGER NOT NULL, FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    ''')

    conn.commit()
    conn.close()


def insert_chunk(file_path: str, chunk: str, embedding: str, project_id: int):
    """
    Inserts a text chunk and its embedding into the database.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('''
                   INSERT INTO chunks (chunk, embedding, project_id)
                   VALUES (?, ?, ?)
                   ''', (chunk, embedding, project_id))

    conn.commit()
    conn.close()


def get_all_chunks(file_path: str):
    """
    Retrieves all text chunks and their embeddings from the database.
    Returns a list of tuples (chunk, embedding, project_id).
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('SELECT chunk, embedding, project_id FROM chunks')
    rows = cursor.fetchall()

    conn.close()
    return rows


def create_project_in_database(file_path: str, project_name: str):
    """
    Creates a new project and returns its ID.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('INSERT INTO projects (project_name) VALUES (?)', (project_name,))
    conn.commit()
    project_id = cursor.lastrowid
    conn.close()

    return project_id

def get_chunks_by_project_id(file_path: str, project_id: int):
    """
    Retrieves all chunks for a specific project.
    Returns a list of tuples (chunk, embedding).
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('SELECT id, chunk, embedding FROM chunks WHERE project_id = ?', (project_id,))
    rows = cursor.fetchall()

    conn.close()
    return rows

def get_projects_from_database(file_path: str):
    """
    Retrieves all projects from the database.
    Returns a list of tuples (id, project_name).
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('SELECT id, project_name FROM projects ORDER BY id DESC')
    rows = cursor.fetchall()

    conn.close()
    return rows


def project_exists(file_path: str, project_id: int) -> bool:
    """
    Checks whether a project exists.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('SELECT 1 FROM projects WHERE id = ? LIMIT 1', (project_id,))
    row = cursor.fetchone()

    conn.close()
    return row is not None


def insert_chat_message(file_path: str, project_id: int, role: str, content: str, best_texts: list[str] | None = None):
    """
    Inserts a chat message for a specific project into the database.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute(
        '''
        INSERT INTO chat_messages (project_id, role, content, best_texts)
        VALUES (?, ?, ?, ?)
        ''',
        (project_id, role, content, json.dumps(best_texts) if best_texts is not None else None)
    )

    conn.commit()
    conn.close()


def get_chat_messages_by_project_id(file_path: str, project_id: int, limit: int | None = 20):
    """
    Retrieves chat messages for a specific project in chronological order.
    Returns a list of tuples (role, content, created_at).
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    query = '''
        SELECT role, content, created_at, best_texts
        FROM chat_messages
        WHERE project_id = ?
        ORDER BY id DESC
    '''
    params = [project_id]

    if limit is not None:
        query += ' LIMIT ?'
        params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    conn.close()
    return list(reversed(rows))


def delete_chat_messages_by_project_id(file_path: str, project_id: int) -> int:
    """
    Deletes all chat messages for a specific project.
    Returns the number of deleted rows.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute(
        '''
        DELETE FROM chat_messages
        WHERE project_id = ?
        ''',
        (project_id,),
    )
    deleted_rows = cursor.rowcount

    conn.commit()
    conn.close()
    return deleted_rows


def delete_project_by_id(file_path: str, project_id: int) -> bool:
    """
    Deletes a project and all related chunks and chat messages.
    Returns True when the project existed and was deleted.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute('SELECT 1 FROM projects WHERE id = ? LIMIT 1', (project_id,))
    project_row = cursor.fetchone()
    if project_row is None:
        conn.close()
        return False

    try:
        cursor.execute('DELETE FROM chat_messages WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM chunks WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM projects WHERE id = ?', (project_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return True


