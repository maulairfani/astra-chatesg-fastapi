from google.cloud import firestore
import datetime
import humps
from typing import List
from scripts.config import settings

from scripts.models import ChatDocument, SessionDocument

class ChatRepository:
    def __init__(self, fsclient: firestore.Client):
        self.fsclient = fsclient
        self.session_path = "chatHistories/{src}/userRecords/{uid}/botSessions"
        self.chat_path = "chatHistories/{src}/userRecords/{uid}/botSessions/{bsid}/botChats"
    
    def create_session_doc(self, item: SessionDocument):
        item.created_at = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            session_path = self.session_path.format(src=item.src, uid=item.uid)
            doc = self.fsclient.collection(session_path).document(item.bsid)
            doc.set(
                humps.camelize(item.model_dump()),
                merge=True
            )
            return item
        except Exception as e:
            raise ValueError(f"Failed to create session document: {e}")
        
    def update_session_doc(self, item: SessionDocument):
        item.updated_at = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            session_path = self.session_path.format(src=item.src, uid=item.uid)
            doc = self.fsclient.collection(session_path).document(item.bsid)
            
            _update = {}
            if item.input_tokens is not None:
                _update['input_tokens'] = firestore.Increment(item.input_tokens)
            if item.output_tokens is not None:
                _update['output_tokens'] = firestore.Increment(item.output_tokens)
            if item.total_tokens is not None:
                _update['total_tokens'] = firestore.Increment(item.total_tokens)
            if item.cost is not None:
                _update['cost'] = firestore.Increment(item.cost)

            for key, value in _update.items():
                setattr(item, key, value)

            item_dump = {key: value for key, value in item.__dict__.items() if value is not None}
            doc.update(humps.camelize(item_dump))
            
            return item
        except Exception as e:
            raise ValueError(f"Failed to update session document: {e}")

    def get_chat_docs(self, src: str, uid: str, bsid: str) -> List[ChatDocument]:
        chat_path = self.chat_path.format(src=src, uid=uid, bsid=bsid)

        try:
            doc_ref = self.fsclient.collection(chat_path)
            docs = doc_ref.order_by('createdAt', direction=firestore.Query.DESCENDING).limit(settings.LIMIT_HISTORY).get()
            docs = [humps.decamelize(doc.to_dict()) for doc in docs]
            docs = [ChatDocument(**doc) for doc in docs]
            return docs
        except Exception as e:
            raise ValueError(f"Failed to get chat documents: {e}")
    
    def create_chat_doc(self, item: ChatDocument):
        item.created_at = datetime.datetime.now(tz=datetime.timezone.utc)

        try:
            chat_path = self.chat_path.format(src=item.src, uid=item.uid, bsid=item.bsid)
            doc = self.fsclient.collection(chat_path).document(item.bcid)
            doc.set(
                humps.camelize(item.model_dump()),
                merge=True
            )
            return item
        except Exception as e:
            raise ValueError(f"Failed to create chat document: {e}")


    