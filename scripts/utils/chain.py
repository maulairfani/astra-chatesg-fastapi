from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from scripts.config import settings

# Model to endpoint mapping untuk model HuggingFace
HUGGINGFACE_MODELS = {
    'climategpt-7b': 'https://vnywjc8vg2jtwu0c.us-east4.gcp.endpoints.huggingface.cloud',
    'gemma-2-27b-it': 'https://sm3rd92c0n31cnxk.us-east4.gcp.endpoints.huggingface.cloud',
    'climategpt-13b': 'https://mcz78eg7btb628gc.us-east4.gcp.endpoints.huggingface.cloud',
    'mistral-7b-instruct': 'https://edk59r9hnxxtlimb.us-east4.gcp.endpoints.huggingface.cloud'
    # Tambahkan model HuggingFace lainnya dan endpoint-nya di sini
}

OPENAI_VALID_MODELS = {'gpt-4o-mini', 'gpt-4o'}
HUGGINGFACE_VALID_MODELS = {'climategpt-7b', 'climategpt-13b', 'Llama-3.1-8B-Instruct', 'gemma-2-27b-it', 'mistral-7b-instruct'}

def init_chain(model: str):
    if model in OPENAI_VALID_MODELS:
        return ChatOpenAI(model=model, temperature=settings.TEMPERATURE)
    elif model in HUGGINGFACE_VALID_MODELS:
        endpoint_url = HUGGINGFACE_MODELS.get(model)
        if not endpoint_url:
            raise ValueError(f"No endpoint URL configured for model {model}")
        return HuggingFaceEndpoint(endpoint_url=endpoint_url, temperature=0.01, max_new_tokens=150)
    else:
        raise ValueError(f"Unsupported model: {model}")
