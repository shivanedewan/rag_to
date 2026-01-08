from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Path, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uuid
from typing import Optional, List, Dict
import PyPDF2
import io
import pytesseract
from PIL import Image
from openai import AsyncOpenAI
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import os
from dotenv import load_dotenv
import json
import asyncio
load_dotenv()
app = FastAPI()
# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024)) # 10MB
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", 60)) # 60 Seconds
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# DB Setup
DATABASE_URL = "sqlite:///./sessions.db"
# connect_args needed for SQLite concurrency
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    history = Column(Text)
    file_context = Column(Text, nullable=True)
Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# VLLM client
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://192.168.17.236:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
# --- Fix 3: Locks ---
session_locks: Dict[str, asyncio.Lock] = {}
def get_session_lock(session_id: str):
    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()
    return session_locks[session_id]
# --- Fix 1: Proper SSE Formatting ---
def format_sse(data: str) -> str:
    """Ensures content ends with double newline for SSE event flushing."""
    if not data:
        return "data: \n\n"
    # Ensure each line starts with "data: " and the whole block ends with \n\n
    return "".join(f"data: {line}\n" for line in data.splitlines()) + "\n\n"
def extract_text_sync(filename: str, contents: bytes) -> str:
    """CPU-bound extraction logic"""
    file_text = ""
    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        file_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    elif filename.endswith(".txt"):
        file_text = contents.decode("utf-8")
    elif filename.endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(io.BytesIO(contents))
        file_text = pytesseract.image_to_string(img)
    return file_text
@app.post("/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    message: Optional[str] = Form(None),
    file: UploadFile = File(None),
    session_id: str = Form(...),
    db: Session = Depends(get_db)
):
    if not message and not file:
        raise HTTPException(status_code=400, detail="Message or file required")
    # --- Lock Acquisition ---
    lock = get_session_lock(session_id)
    await lock.acquire()
    try:
        # 1. Prepare Session (uses request-scoped DB)
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            session = ChatSession(session_id=session_id, history="[]")
            db.add(session)
            db.commit()
            db.refresh(session)
       
        history: List[Dict[str, str]] = json.loads(session.history)
        # 2. Handle File (with Size Limit)
        if file:
            filename = file.filename.lower()
            if not filename.endswith((".pdf", ".txt", ".jpg", ".jpeg", ".png")):
                raise HTTPException(status_code=400, detail="Unsupported file type.")
           
            # Fix 5: Read with size limit check
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            try:
                # Offload to threadpool
                file_text = await run_in_threadpool(extract_text_sync, filename, contents)
                session.file_context = file_text
                db.commit()
            except Exception as e:
                logger.error(f"File processing error: {e}")
                raise HTTPException(status_code=400, detail="Error processing file")
        # 3. Update History & Messages
        user_message = message or ""
        history.append({"role": "user", "content": user_message})
        messages = []
        if session.file_context:
            messages.append({
                "role": "system",
                "content": f"Use the following document to answer:\n{session.file_context}"
            })
        messages.extend(history)
        print(messages)
        # 4. Define Generator
        async def generate():
            nonlocal history
            assistant_response = ""
           
            try:
                # Fix 6 & 4: Timeout and Stream Validation
                # We wrap the creation in a timeout to prevent hanging forever
                stream = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        stream=True
                    ),
                    timeout=10.0 # Timeout for initial connection
                )
                # Iterate through stream with a timeout for each chunk
                async for chunk in stream:
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected: {session_id}")
                        return
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        assistant_response += content
                        yield format_sse(content)
                yield "event: done\ndata: [DONE]\n\n"
                if await request.is_disconnected():
                    return
                # Fix 2: Independent DB Session for persistence
                # We use a fresh SessionLocal() because the outer 'db' dependency
                # might be closed by FastAPI once the response started streaming.
                save_db = SessionLocal()
                try:
                    # Re-fetch session to attach to this new DB session
                    s = save_db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
                    if s:
                        # Append assistant response to history
                        history.append({"role": "assistant", "content": assistant_response})
                        s.history = json.dumps(history)
                        save_db.commit()
                except Exception as e:
                    logger.error(f"Failed to save history: {e}")
                    save_db.rollback()
                finally:
                    save_db.close()
            except asyncio.TimeoutError:
                yield format_sse("Error: Generation timed out.")
                yield "event: done\ndata: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Generation error: {e}")
                yield format_sse(f"Error: {str(e)}")
                yield "event: done\ndata: [DONE]\n\n"
            finally:
                # Fix 3: Release lock in finally block of generator
                if lock.locked():
                    lock.release()
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        # Release lock if we crash before streaming starts
        if lock.locked():
            lock.release()
        logger.error(f"Setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/new_session")
async def new_session(db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    session = ChatSession(session_id=session_id, history="[]")
    db.add(session)
    db.commit()
    return {"session_id": session_id}
@app.get("/session/{session_id}")
async def get_session(session_id: str = Path(...), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": json.loads(session.history), "file_context": session.file_context}