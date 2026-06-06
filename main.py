import json
import uuid
import os
import asyncio
from typing import cast
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionUserMessageParam

import src.database_utils as database_utils
from src.embedding_utils import cosine_similarity
from src.llm_utils import ask_llm
from src.requests_utils import create_embedding, EmbeddingServiceError
from src.utils import get_all_text_from_pdf, join_sentences_into_chunks, split_text_into_by_sentences

app = FastAPI()

# Global dictionary to track upload progress by project_id
upload_progress = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", # React dev server
        "http://127.0.0.1:3000", # React dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],  # or ["GET", "POST", ...]
    allow_headers=["*"],
)

PROJECT_DIR = './projects'
DATABASE_NAME = 'embeddings.db'
DATABASE_FILE = os.path.join(PROJECT_DIR, DATABASE_NAME)

os.makedirs(PROJECT_DIR, exist_ok=True)
database_utils.create_database(DATABASE_FILE)

# POST /projects - Create new project
@app.post("/projects")
async def create_project(project_name: str):
    project_id = database_utils.create_project_in_database(DATABASE_FILE, project_name)
    return JSONResponse(content={"project_id": project_id, "message": "Project created successfully."})

# GET /projects - Get all project IDs and names
@app.get("/projects")
async def get_projects():
    projects = database_utils.get_projects_from_database(DATABASE_FILE)
    return JSONResponse(content={"projects": [{"project_id": project_id, "project_name": project_name} for project_id, project_name in projects]})

# POST /projects/{project_id}/upload - Upload PDF and extract chunks
@app.post("/projects/{project_id}/upload")
async def upload_pdf(project_id: int, file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    file_location = os.path.join(PROJECT_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Initialize progress tracking
        upload_progress[project_id] = {
            "status": "processing",
            "progress": 0,
            "current": 0,
            "total": 0,
            "stage": "extracting_text"
        }

        text = get_all_text_from_pdf(file_location)
        chunks = join_sentences_into_chunks(split_text_into_by_sentences(text), chunk_size=400, overlap_percentage=0.5)
        
        upload_progress[project_id]["total"] = len(chunks)
        upload_progress[project_id]["stage"] = "generating_embeddings"

        for index, chunk in enumerate(chunks):
            embedding = create_embedding(chunk)
            database_utils.insert_chunk(DATABASE_FILE, chunk, str(embedding), project_id)
            
            # Update progress
            upload_progress[project_id]["current"] = index + 1
            upload_progress[project_id]["progress"] = int((index + 1) / len(chunks) * 100)
            await asyncio.sleep(0)  # Allow other tasks to run

        upload_progress[project_id]["status"] = "completed"
        upload_progress[project_id]["progress"] = 100
        return JSONResponse(content={"message": f"File uploaded and processed successfully. {len(chunks)} chunks created."})
    except EmbeddingServiceError as exc:
        upload_progress[project_id]["status"] = "error"
        upload_progress[project_id]["error"] = str(exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)  # Clean up the uploaded file even on error

# GET /projects/{project_id}/chunks - Get all chunks
@app.get("/projects/{project_id}/chunks")
async def get_chunks(project_id: int):
    chunks = database_utils.get_chunks_by_project_id(DATABASE_FILE, project_id)
    return JSONResponse(content={"chunks": [{"id": id, "chunk": chunk, "embedding": json.loads(embedding)} for id, chunk, embedding in chunks]})


# GET /projects/{project_id}/chat/history - Get chat history
@app.get("/projects/{project_id}/chat/history")
async def get_chat_history(project_id: int):
    if not database_utils.project_exists(DATABASE_FILE, project_id):
        raise HTTPException(status_code=404, detail="Project not found.")

    messages = database_utils.get_chat_messages_by_project_id(DATABASE_FILE, project_id, limit=None)
    return JSONResponse(content={"chat_history": [{"role": role, "content": content, "created_at": created_at} for role, content, created_at in messages]})


# GET /projects/{project_id}/chat/history/latest - Get latest chat history messages
@app.get("/projects/{project_id}/chat/history/latest")
async def get_latest_chat_history(project_id: int, limit: int = 20):
    if not database_utils.project_exists(DATABASE_FILE, project_id):
        raise HTTPException(status_code=404, detail="Project not found.")

    if limit <= 0:
        raise HTTPException(status_code=400, detail="Limit must be greater than 0.")

    messages = database_utils.get_chat_messages_by_project_id(DATABASE_FILE, project_id, limit=limit)
    return JSONResponse(content={
        "chat_history": [
            {"role": role, "content": content, "created_at": created_at, "best_texts": json.loads(best_texts) if best_texts else None}
            for role, content, created_at, best_texts in messages
        ],
        "limit": limit,
    })


# DELETE /projects/{project_id}/chat/history - Delete chat history for a project
@app.delete("/projects/{project_id}/chat/history")
async def delete_chat_history(project_id: int):
    if not database_utils.project_exists(DATABASE_FILE, project_id):
        raise HTTPException(status_code=404, detail="Project not found.")

    deleted_messages = database_utils.delete_chat_messages_by_project_id(DATABASE_FILE, project_id)
    return JSONResponse(content={
        "message": "Chat history deleted successfully.",
        "deleted_messages": deleted_messages,
    })

# GET /projects/{project_id}/upload/progress - Get upload progress
@app.get("/projects/{project_id}/upload/progress")
async def get_upload_progress(project_id: int):
    if project_id in upload_progress:
        return JSONResponse(content=upload_progress[project_id])
    else:
        return JSONResponse(content={"status": "not_started", "progress": 0, "current": 0, "total": 0})


# DELETE /projects/{project_id} - Delete project with all embeddings and chat history
@app.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    deleted = database_utils.delete_project_by_id(DATABASE_FILE, project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found.")

    upload_progress.pop(project_id, None)

    return JSONResponse(content={
        "message": "Project deleted successfully.",
        "project_id": project_id,
    })

# GET /projects/{project_id}/ask - Ask a question and get an answer from LLM
@app.get("/projects/{project_id}/ask")
async def ask_question(project_id: int, query: str):
    if not database_utils.project_exists(DATABASE_FILE, project_id):
        raise HTTPException(status_code=404, detail="Project not found.")

    query_embedding = create_embedding(query)
    chunks_from_db = database_utils.get_chunks_by_project_id(DATABASE_FILE, project_id)
    chat_history_rows = database_utils.get_chat_messages_by_project_id(DATABASE_FILE, project_id, limit=20)
    chat_history: list[ChatCompletionMessageParam] = []
    for role, content, _created_at, best_texts in chat_history_rows:
        if role == "assistant":
            chat_history.append(cast(ChatCompletionAssistantMessageParam, {"role": "assistant", "content": content}))
        else:
            chat_history.append(cast(ChatCompletionUserMessageParam, {"role": "user", "content": content}))

    results = []
    for chunk_from_db in chunks_from_db:
        score = cosine_similarity(query_embedding, json.loads(chunk_from_db[2]))
        results.append({
            "score": score,
            "chunk_text": chunk_from_db[1],
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    n = 10
    best_chunks = results[:n]
    best_texts = [result['chunk_text'] for result in best_chunks]
    answer = ask_llm(
        query,
        best_texts,
        chat_history=chat_history,
    )

    database_utils.insert_chat_message(DATABASE_FILE, project_id, "user", query)
    database_utils.insert_chat_message(DATABASE_FILE, project_id, "assistant", answer, best_texts)

    return JSONResponse(content={
        "answer": answer,
        "chat_history": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ],
        best_texts: best_texts,
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

