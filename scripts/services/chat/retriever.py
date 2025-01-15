from pydantic import BaseModel
from langchain_core.documents import Document
from typing import List
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import os
import yaml
import re
from google.cloud import firestore

from scripts.schemas import (
    RetrieverRequest,
    RetrieverResponse
)
from scripts.repositories import ChatRepository
from scripts.config import settings
from scripts.utils import init_chain
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
class RelevantIndicators(BaseModel):
    indicator: str = Field(description="Most relevant indicator towards the question")

class Retriever:
    def __init__(self, fsclient: firestore.Client):
        self.vector_store = self._load_vector_store()
        self.repo = ChatRepository(fsclient=fsclient)
        self.prompts = self._load_prompts()
        self.embedding_function = OpenAIEmbeddings(model=settings.EMBEDDINGS)
        self.valid_indicators = [
            '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10',
            '2-11', '2-12', '2-13', '2-14', '2-15', '2-16', '2-17', '2-18', '2-19',
            '2-20', '2-21', '2-22', '2-23', '2-24', '2-25', '2-26', '2-27', '2-28',
            '2-29', '2-30', '3-1', '3-2', '3-3', '201-1', '201-2', '201-3', '201-4',
            '202-1', '202-2', '203-1', '203-2', '204-1', '205-1', '205-2', '205-3',
            '206-1', '207-1', '207-2', '207-3', '207-4', '301-1', '301-2', '301-3',
            '302-1', '302-2', '302-3', '302-4', '302-5', '303-1', '303-2', '303-3',
            '303-5', '304-1', '304-2', '304-3', '304-4', '305-1', '305-2', '305-3',
            '305-4', '305-5', '305-6', '305-7', '306-1', '306-2', '306-3', '306-4',
            '306-5', '308-1', '308-2', '401-1', '401-2', '401-3', '402-1', '403-1',
            '403-2', '403-3', '403-4', '403-5', '403-6', '403-7', '403-8', '403-9',
            '403-10', '404-1', '404-2', '404-3', '405-1', '405-2', '406-1', '407-1',
            '408-1', '409-1', '410-1', '411-1', '413-1', '413-2', '414-1', '414-2',
            '415-1', '416-1', '416-2', '417-1', '417-2', '417-3', '418-1'
        ]

    async def get_relevant_contents(self, request: RetrieverRequest):
        response = RetrieverResponse()

        # Similarity-based retrieval
        if request.mode == 'similarity' or request.mode == 'combination':
            docs, metadata = self.similarity_retrieve(request)
            response.contents.extend(docs)
            response.metadata.update(metadata)

        # Indicator classification-based retrieval
        if request.mode == 'indicator-cls' or request.mode == 'combination':
            docs, metadata = self.indicator_cls_retrieve(request)
            response.contents.extend(docs)
            response.metadata.update(metadata)

        # get metadata
        company_data = self.repo.get_data_by_company_name(request.filter['company'], request.filter['year'])
        response.metadata.update({
            "url": company_data.get("url")
        })

        return response

    def similarity_retrieve(self, request: RetrieverRequest):
        _filter = {}
        for f in request.filter:
            if request.filter[f] != None:
                _filter[f] = {"$eq": request.filter[f]}
                
        vector = self.embedding_function.embed_query(request.query)
        raw_contents = self.vector_store.query(
            vector=vector,
            filter=_filter,
            top_k=request.top_k if request.top_k != None else settings.TOP_K,
            include_metadata=True
        )
        
        contents = []
        for content in raw_contents['matches']:
            metadata = content['metadata']
            page_content = metadata.pop('text')
            contents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        return contents, {}

    def indicator_cls_retrieve(self, request: RetrieverRequest):
        llm = init_chain(request.model)
        prompt = self.prompts['indicator_cls_prompt'].format(question=request.query)
        
        indicators = []
        # Structurize LLM
        try:
            sllm = llm.with_structured_output(RelevantIndicators)
        except:
            sllm = llm

        try:
            raw_indicators = sllm.invoke(prompt)
            if isinstance(raw_indicators, RelevantIndicators):
                for key, value in raw_indicators.model_dump().items():
                    indi = self._get_indicator(value)
                    indicators.extend(indi)
            else:
                indi = self._get_indicator(raw_indicators)
                indicators.extend(indi)
                indicators = indicators[:2]

        except Exception as e:
            print(f"Failed to classify indicator: {e}")
        
        company_data = self.repo.get_data_by_company_name(request.filter['company'], request.filter['year'])
        cid = company_data.get('cid')
        page_ids = self.repo.get_page_ids_by_gri(cid, indicators)
        contents = self._get_content_by_pids(page_ids)

        # Prepare metadata
        metadata = {'indicators': indicators, 'raw_indicators': raw_indicators}

        return contents, metadata

    def _get_indicator(self, content: str) -> List[str]:
        pattern = r'\b\d{1,3}-\d{1,2}(?=\D|$)'
        found_codes = re.findall(pattern, content)
        disclosed_indicators = [code for code in found_codes if code in self.valid_indicators]
        disclosed_indicators = list(set(disclosed_indicators))
        return disclosed_indicators

    def _get_content_by_pids(self, page_ids: list[dict]):
        contents = []
        for data in page_ids:
            if data['page_ids'] == None:
                continue

            vectors = self.vector_store.fetch(data['page_ids']).to_dict()['vectors']
            
            for pid in list(vectors.keys()):
                page_content = ""
                metadata = {}
                for key, value in vectors[pid]['metadata'].items():
                    if key == 'text':
                        page_content = value
                    else:
                        metadata[key] = value
                
                if len(page_content) > 0:
                    contents.append(
                        Document(
                            page_content=page_content,
                            metadata=metadata
                        )
                    )
        return contents
                    
    def _load_vector_store(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(settings.INDEX)
        return index
    
    def _load_prompts(self):
        with open("scripts\services\chat\prompts.yml", "r") as file:
            data = yaml.safe_load(file)
        return data
    
    