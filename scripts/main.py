from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from firebase_admin import delete_app

from scripts.utils import get_firestore_client
from scripts.config import settings
from scripts.routers.api import router
from dotenv import load_dotenv
load_dotenv()

def startup(app: FastAPI):
    # Inisialisasi Firestore Client
    app.fsclient = get_firestore_client()

    # Inisialisasi Pinecone Vector Store
    # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # app.vector_store = PineconeVectorStore(
    #     index=pc.Index(settings.INDEX),
    #     embedding=OpenAIEmbeddings(model=settings.EMBEDDINGS)
    # )
    app.vector_store = 'test'

def shutdown(app: FastAPI):
    # Tutup Firebase app jika ada
    if hasattr(app, "fa_app"):
        delete_app(app.fa_app)

app = FastAPI()
startup(app)

origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "OK"}

app.include_router(router)