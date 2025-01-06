import json
import os
from firebase_admin import firestore as admin_firestore, credentials, initialize_app, _apps

def get_firestore_client():
    if not _apps:  # Initialize Firebase only if it hasn't been initialized
        cred = credentials.Certificate(json.loads(os.getenv("CRED_FIREBASE")))
        initialize_app(credential=cred)
    return admin_firestore.client()
