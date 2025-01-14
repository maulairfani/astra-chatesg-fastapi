from pinecone import Pinecone
from google.cloud import firestore
import uuid
import json
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import time
from pydantic import BaseModel, Field
import yaml

from scripts.repositories import ChatRepository
from scripts.schemas import (
    ChatRequest, 
    ChatResponse, 
    SourceItem,
    RetrieverRequest,
    RetrieverResponse
)
from scripts.models import SessionDocument, ChatDocument
from scripts.services.chat.retriever import Retriever
from scripts.config import settings

class ChatService:
    def __init__(self, vector_store: str, fsclient: firestore.Client, uid: str, model: str):
        self.vector_store = vector_store
        self.prompts = self._load_prompts()
        self.repo = ChatRepository(fsclient=fsclient)
        self.retriever = Retriever(model, fsclient)
        self.uid = uid

    async def create(self, request: ChatRequest):
        # Initialize response
        response = ChatResponse(
            bsid=request.bsid if request.bsid != None else uuid.uuid4().hex[:20],
            bcid=str(uuid.uuid4()),
        )

        # Load chat history
        chat_history = []
        if request.bsid != None:
            response.tools.append("Mengambil histori percakapan")
            yield json.dumps(response.model_dump()) + "\n"
            chat_history = self._load_chat_history(request.src, self.uid, request.bsid)
        
        # Initialize chain
        chain = self._init_chain(request.model)

        # Build input
        system_messages = [
            SystemMessage(content=self.prompts['rag_prompt'])
        ]

        user_messages = []
        user_messages_str = ""
        for input in request.inputs:
            if input.type == "text":
                if input.role == "user":
                    user_messages.append(
                        HumanMessage(content=input.content)
                    )
                    user_messages_str += input.content + "\n\n"
                else:
                    raise ValueError("Role other than user is denied")
            else:
                raise NotImplementedError("Input other than text is not implemented")
        
        # Init Retriever
        retriever_request = RetrieverRequest(
            query=user_messages_str, 
            filter={"company": request.company, "year": request.year}
        )
        response.tools.append("Mengambil informasi relevan dari laporan keberlanjutan")
        yield json.dumps(response.model_dump()) + "\n"
        retriever_response = self.retriever.get_relevant_contents(retriever_request)
        context_messages = [
            SystemMessage(content=self.prompts['context_prompt'].format(contexts=retriever_response.contents, company=request.company, year=request.year))
        ]
        print(retriever_response.metadata)

        # Start streaming response
        messages = [*chat_history, *system_messages, *context_messages, *user_messages]
        run_config = {"run_name": "Answer Generation", "run_id": response.bcid}
        for chunk in chain.stream(messages, config=run_config):
            if len(response.responses) == 0:
                response.responses.append("")
            
            response.responses[-1] += chunk
            
            yield json.dumps(response.model_dump()) + "\n"

        # Return sources
        response.sources = [
            SourceItem(
                url=retriever_response.metadata.get('url'),
                pages=[int(doc.metadata['page']) for doc in retriever_response.contents],
                related_indicators=retriever_response.metadata.get('indicators')
            ).model_dump()
        ]
        response.done = True
        yield json.dumps(response.model_dump()) + "\n"

        if request.bsid == None:
            title = self._create_title(request.inputs, request.model)
            # Save session doc
            session_result = self._create_or_update_session_doc(request, response, title)
        else:
            session_result = self._create_or_update_session_doc(request, response)
            

        # Save chat doc
        chat_result = self._create_chat_doc(request, response)

    def _load_chat_history(self, src: str, uid: str, bsid: str):
        docs = self.repo.get_chat_docs(src, uid, bsid)

        chat_history = []
        for doc in docs:
            for input in doc.inputs:
                chat_history.append(
                    HumanMessage(content=input.content)
                )
            for output in doc.outputs:
                chat_history.append(
                    AIMessage(content=output.content)
                )

        return chat_history

    def _init_chain(self, model: str):
        huggingface_valid_model = ['climategpt-7b', 'Llama-3.1-8B-Instruct', 'Llama-3.2-11B-Vision']
        openai_valid_model = ['gpt-4o-mini', 'gpt-4o']
        if model in openai_valid_model:
            chain = ChatOpenAI(model=model, temperature=settings.TEMPERATURE)
        elif model in huggingface_valid_model:
            if model == 'climategpt-7b':
                endpoint_url = 'https://vnywjc8vg2jtwu0c.us-east4.gcp.endpoints.huggingface.cloud'
            elif model == 'Llama-3.1-8B-Instruct':
                # endpoint_url = 'https://ud5os6horbh2229p.us-east-1.aws.endpoints.huggingface.cloud'
                endpoint_url = 'https://dcds09i4mh9vmwh7.us-east4.gcp.endpoints.huggingface.cloud'
            chain = HuggingFaceEndpoint(endpoint_url=endpoint_url, temperature=settings.TEMPERATURE)
        return chain
    
    def _create_or_update_session_doc(self, request: ChatRequest, response: ChatResponse, title: str | None = None):
        session_doc = SessionDocument(
            version_no=1,
            src=request.src,
            uid=self.uid,
            bsid=response.bsid,
            is_deleted=False,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost=0,
            title=title,
            suggestions=None
        )
        if request.bsid == None:
            session_result = self.repo.create_session_doc(session_doc)
        else:
            session_result = self.repo.update_session_doc(session_doc)
        
        return session_result
    
    def _create_chat_doc(self, request: ChatRequest, response: ChatResponse):
        chat_doc = ChatDocument(
            version_no=1,
            src=request.src,
            uid=self.uid,
            bsid=response.bsid,
            bcid=response.bcid,
            is_deleted=False,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost=0,
            inputs=[item.model_dump() for item in request.inputs],
            outputs=[{"type": "text", "role": "assistant", "content": res} for res in response.responses],
            start_time=None,
            latency_ms=None,
            first_token_ms=None,
            report=None,
            sources=response.sources,
            metadata=None
        )
        chat_result = self.repo.create_chat_doc(chat_doc)
        return chat_result

    def _create_title(self, inputs, model):
        # Load prompt
        prompt = self.prompts['create_title']

        class SessionTitle(BaseModel):
            title: str = Field(None, title="Judul sesi")

        llm = ChatOpenAI(model='gpt-4o-mini', temperature=settings.TEMPERATURE)
        llm = llm.with_structured_output(SessionTitle)

        prompt = prompt.format(inputs=inputs)
        response = llm.invoke(prompt, config={"run_name": "Create Title"})

        return response.title
    
    def _load_prompts(self):
        with open("scripts\services\chat\prompts.yml", "r") as file:
            data = yaml.safe_load(file)
        return data
