from google.cloud import firestore
import datetime
import humps
from typing import List

from scripts.config import settings
from scripts.models import ChatDocument, SessionDocument

import warnings
warnings.filterwarnings("ignore")

class ChatRepository:
    def __init__(self, fsclient: firestore.Client):
        self.fsclient = fsclient
        self.session_path = "chatHistories/{src}/userRecords/{uid}/botSessions"
        self.chat_path = "chatHistories/{src}/userRecords/{uid}/botSessions/{bsid}/botChats"
        self.sr_path = "sustainabilityReports/{cid}"
    
    def create_session_doc(self, item: SessionDocument):
        item.created_at = datetime.datetime.now(tz=datetime.timezone.utc)
        item.updated_at = datetime.datetime.now(tz=datetime.timezone.utc)

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
            col_ref = self.fsclient.collection(chat_path)
            docs = col_ref.order_by('createdAt', direction=firestore.Query.DESCENDING).limit(settings.LIMIT_HISTORY).get()
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

    def get_data_by_company_name(self, company_name: str, year: int):
        sr_path = self.sr_path.replace("/{cid}", "")

        try:
            # Referensi ke koleksi
            col_ref = self.fsclient.collection(sr_path)
            
            # Filter dokumen berdasarkan company_name dan year
            query = col_ref.where("company", "==", company_name).where("year", "==", year)
            docs = query.get()
            
            # Pastikan hanya satu dokumen yang dikembalikan
            if len(docs) == 1:
                doc_data = docs[0].to_dict()  # Convert dokumen ke dictionary
                return doc_data  # Ambil nilai 'cid'
            elif len(docs) > 1:
                raise ValueError(f"Multiple documents found for company '{company_name}' and year '{year}'. Expected only one.")
            else:
                raise ValueError(f"No document found for company '{company_name}' and year '{year}'.")
        
        except Exception as e:
            raise RuntimeError(f"An error occurred while fetching the CID: {e}")
            
    def get_page_ids_by_gri(self, cid: str, gri_codes: list[str]):
        sr_path = self.sr_path.format(cid=cid)

        try:
            doc_ref = self.fsclient.document(sr_path)
            doc_data = doc_ref.get().to_dict()
            disclosed_gri = doc_data.get('disclosedGri')
            
            page_ids = []
            for code in gri_codes:
                page_ids.append({
                    "code": code,
                    "page_ids": disclosed_gri.get(code)
                })
            return page_ids
        except Exception as e:
            raise ValueError(f"Failed to get page ids: {e}")

    def update_user_feedback(self, item: ChatDocument):
        chat_path = self.chat_path.format(src=item.src, uid=item.uid, bsid=item.bsid)

        try:
            _update = {}
            if item.thumbs_up == True:
                _update['thumbs_up'] = True
            elif item.thumbs_up == False:
                _update['thumbs_up'] = False
                _update['report'] = item.report
            elif item.thumbs_up == None:
                _update['thumbs_up'] = None
                _update['report'] = None
            else:
                raise ValueError("Request doesn't suitable to update feedback")
            
            doc = self.fsclient.collection(chat_path).document(item.bcid)
            doc.update(humps.camelize(_update))
        except Exception as e:
            raise ValueError(f"Failed to update user feedback: {e}")
        
        return _update