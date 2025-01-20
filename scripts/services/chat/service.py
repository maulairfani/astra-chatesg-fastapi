from google.cloud import firestore
import uuid
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk
import time
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import yaml
import os

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
from scripts.utils import init_chain

class ChatService:
    # Cache prompts sebagai variabel kelas
    _prompts_cache: Optional[Dict] = None

    def __init__(self, fsclient: firestore.Client, uid: str, model: str):
        self.prompts = self._load_prompts()
        self.repo = ChatRepository(fsclient=fsclient)
        self.uid = uid
        self.chain = init_chain(model)
        self.retriever = Retriever(fsclient)

    @classmethod
    def _load_prompts(cls) -> dict:
        if cls._prompts_cache is None:
            prompt_path = os.path.join("scripts", "services", "chat", "prompts.yml")
            with open(prompt_path, "r", encoding="utf-8") as file:
                cls._prompts_cache = yaml.safe_load(file)
        return cls._prompts_cache

    def create(self, request: ChatRequest):
        # Initialize response
        response = ChatResponse(
            bsid=request.bsid or uuid.uuid4().hex[:20],
            bcid=str(uuid.uuid4()),
        )

        # Load prompt template
        prompt_template: str = self.prompts.get('rag_prompt')
        if not prompt_template:
            raise ValueError("Prompt template 'rag_prompt' not found.")

        # Load chat history if bsid is provided
        chat_history = ""
        if request.bsid:
            response.tools.append("Mengambil histori percakapan")
            yield json.dumps(response.model_dump()) + "\n"
            chat_history = self._load_chat_history(request.src, self.uid, request.bsid)

        # Process user inputs
        user_messages = self._aggregate_user_messages(request.inputs)

        # Retrieve Context
        if request.reference_contexts:
            # For evaluation purpose only
            retriever_response = RetrieverResponse(
                metadata={
                    'indicators': [request.reference_contexts]
                }
            )

            company_data = self.repo.get_data_by_company_name(request.company, request.year)
            cid = company_data.get('cid')
            retriever_response.metadata.update({
                "url": company_data.get("url")
            })

            # get contents
            page_ids = self.repo.get_page_ids_by_gri(cid, [request.reference_contexts])
            retriever_response.contents = self.retriever._get_content_by_pids(page_ids)
            
            # aggregate contexts
            contexts = self._aggregate_contexts(retriever_response.contents)

        else:
            retriever_request = RetrieverRequest(
                query=user_messages,
                filter={"company": request.company, "year": request.year},
                mode=request.retriever,
                top_k=request.top_k,
                model=request.model,
            )
            response.tools.append("Mengambil informasi relevan dari laporan keberlanjutan")
            yield json.dumps(response.model_dump()) + "\n"
            retriever_response = self.retriever.get_relevant_contents(retriever_request)

            # Aggregate contexts
            contexts = self._aggregate_contexts(retriever_response.contents)

        if len(contexts) == 0:
            contexts = "Tidak ada informasi terkait dalam laporan keberlanjutan atau informasi tersebut tidak diungkap dalam laporan keberlanjutan"
        
        if len(chat_history) == 0:
            chat_history = "Histori percakapan tidak tersedia"

        # Format prompt
        message = prompt_template.format(
            chat_history=chat_history,
            contexts=contexts,
            question=user_messages,
            company=request.company,
            year=request.year
        )

        # Start streaming response
        run_config = {
            "run_name": "Answer Generation",
            "run_id": response.bcid,
            "metadata": {
                "configurable": {
                    "session_id": response.bsid,
                    "user_id": self.uid
                }
            }
        }

        try:
            if not response.responses:
                response.responses.append("")
            for chunk in self.chain.stream(message, config=run_config):
                if isinstance(chunk, AIMessageChunk):
                    content = chunk.content
                elif isinstance(chunk, str):
                    content = chunk
                else:
                    raise ValueError(f"Unexpected type of response chunk: {type(chunk)}")
                
                # Clean the content by removing special tokens
                cleaned_content = self._clean_response(content)
                
                response.responses[-1] += cleaned_content
                yield json.dumps(response.model_dump()) + "\n"
        except Exception as e:
            error_message = str(e)
            if "max_new_tokens" in error_message:
                response.responses[-1] += self._clean_response(error_message)
                yield json.dumps(response.model_dump()) + "\n"
            else:
                # Handle or log other exceptions as needed
                response.done = True
                yield json.dumps(response.model_dump()) + "\n"
                raise

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

        # Save session and chat documents
        title = None
        if not request.bsid and request.src == "app":
            title = self._create_title(request.inputs)
        
        self._create_or_update_session_doc(request, response, title)
        self._create_chat_doc(request, response)

    def _aggregate_user_messages(self, inputs: List) -> str:
        messages = []
        for input_item in inputs:
            if input_item.type != "text":
                raise NotImplementedError(f"Input type '{input_item.type}' is not implemented")
            if input_item.role != "user":
                raise ValueError(f"Role '{input_item.role}' is denied")
            messages.append(input_item.content.strip())
        return "\n\n".join(messages)

    def _aggregate_contexts(self, contents: List) -> str:
        context_list = [
            f"Laporan Keberlanjutan {content.metadata.get('company')} tahun {int(content.metadata.get('year'))}\n"
            f"Halaman: {int(content.metadata.get('page'))}\nKonten: {content.page_content.strip()}"
            for content in contents
        ]
        return "\n\n".join(context_list)

    def _load_chat_history(self, src: str, uid: str, bsid: str) -> str:
        docs = self.repo.get_chat_docs(src, uid, bsid)
        history = []
        for doc in docs:
            user_contents = " ".join(input_item.content.strip() for input_item in doc.inputs)
            assistant_contents = " ".join(output.content.strip() for output in doc.outputs)
            history.append(f"User: {user_contents}\nChatESG: {assistant_contents}")
        return "\n\n".join(history)

    def _create_or_update_session_doc(self, request: ChatRequest, response: ChatResponse, title: Optional[str] = None):
        session_doc = SessionDocument(
            version_no=1,
            src=request.src,
            uid=self.uid,
            bsid=response.bsid,
            is_deleted=False,
            input_tokens=0,  # These should be calculated based on actual token usage
            output_tokens=0,
            total_tokens=0,
            cost=0,
            title=title,
            suggestions=None
        )
        if not request.bsid:
            self.repo.create_session_doc(session_doc)
        else:
            self.repo.update_session_doc(session_doc)

    def _create_chat_doc(self, request: ChatRequest, response: ChatResponse):
        chat_doc = ChatDocument(
            version_no=1,
            src=request.src,
            uid=self.uid,
            bsid=response.bsid,
            bcid=response.bcid,
            is_deleted=False,
            input_tokens=0,  # These should be calculated based on actual token usage
            output_tokens=0,
            total_tokens=0,
            cost=0,
            inputs=[item.model_dump() for item in request.inputs],
            outputs=[{"type": "text", "role": "assistant", "content": res} for res in response.responses],
            start_time=None,  # Should capture actual start time
            latency_ms=None,   # Should capture actual latency
            first_token_ms=None,  # Should capture actual token time
            report=None,
            sources=response.sources,
            metadata=None
        )
        self.repo.create_chat_doc(chat_doc)

    def _create_title(self, inputs: List) -> Optional[str]:
        prompt_template = self.prompts.get('create_title')
        if not prompt_template:
            raise ValueError("Prompt template 'create_title' not found.")

        class SessionTitle(BaseModel):
            title: Optional[str] = Field(None, title="Judul sesi")

        llm = ChatOpenAI(model='gpt-4o-mini', temperature=settings.TEMPERATURE)
        llm = llm.with_structured_output(SessionTitle)

        user_inputs = "\n".join(input_item.content.strip() for input_item in inputs)
        prompt = prompt_template.format(inputs=user_inputs)
        response = llm.invoke(prompt, config={"run_name": "Create Title"})

        return response.title

    def _clean_response(self, text: str) -> str:
        # Define the special tokens to remove
        special_tokens = ['<end_of_turn>', '<eos>', '<\s>']
        for token in special_tokens:
            text = text.replace(token, '')
        return text