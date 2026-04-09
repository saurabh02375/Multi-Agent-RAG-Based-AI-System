from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from backend.api import endpoints, chat
from backend.services.rag import get_retriever
import os

app = FastAPI(title="RAG Backend")

# Mount Static Files
# We use absolute path to be safe, assuming running from root of repo
# os.getcwd() should be the root where 'backend' folder is.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include routers
app.include_router(endpoints.router, prefix="/api")
app.include_router(chat.router, prefix="/api/chat")


@app.on_event("startup")
def init_rag_retriever():
    """
    Build the retriever at startup so the first chat
    request doesn't have to wait for PDF loading + embedding.
    """
    get_retriever()


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
