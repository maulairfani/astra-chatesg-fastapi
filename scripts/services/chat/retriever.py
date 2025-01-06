from pydantic import BaseModel
from langchain_core.documents import Document
from typing import List
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import os

from scripts.schemas import (
    RetrieverRequest,
    RetrieverResponse
)
from scripts.config import settings
from dotenv import load_dotenv
load_dotenv()

class Retriever:
    def __init__(self):
        self.vector_store = self._load_vector_store()
        self.embedding_function = OpenAIEmbeddings(model=settings.EMBEDDINGS)

    def get_relevant_contents(self, request: RetrieverRequest):
        _filter = {}
        for f in request.filter:
            if request.filter[f] != None:
                _filter[f] = {"$eq": request.filter[f]}
                
        vector = self.embedding_function.embed_query(request.query)
        contents = self.vector_store.query(
            vector=vector,
            filter=_filter,
            top_k=settings.TOP_K,
            include_metadata=True
        )
        
        docs = []
        for content in contents['matches']:
            metadata = content['metadata']
            page_content = metadata.pop('text')    
            docs.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        return RetrieverResponse(contents=docs)

    def _load_vector_store(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(settings.INDEX)
        return index
