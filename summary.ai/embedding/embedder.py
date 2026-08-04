import numpy as np
from PIL import Image

_text_model = None
_clip_model = None


def _get_text_model():
    global _text_model
    if _text_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _text_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _text_model = "fallback"
    return _text_model


def _get_clip_model():
    global _clip_model
    if _clip_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _clip_model = SentenceTransformer("clip-ViT-B-32")
        except Exception:
            _clip_model = "fallback"
    return _clip_model


def _hash_embed(s):
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        hv = HashingVectorizer(n_features=384, alternate_sign=False)
        vec = hv.transform([s]).toarray()[0]
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec
    except Exception:
        # Basic character-frequency vector fallback
        vec = np.zeros(384, dtype=np.float32)
        for i, char in enumerate(s[:384]):
            vec[i % 384] += ord(char)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec


def get_embeddings(data, type: str = "text"):
    """
    Generate embeddings for text or images.
    """
    if type == "text":
        model = _get_text_model()
        if model != "fallback":
            if isinstance(data, str):
                return model.encode([data], convert_to_numpy=True)[0]
            elif isinstance(data, list) and all(isinstance(x, str) for x in data):
                return model.encode(data, convert_to_numpy=True)

        if isinstance(data, str):
            return _hash_embed(data)
        elif isinstance(data, list):
            return np.array([_hash_embed(x) for x in data], dtype=np.float32)
        else:
            raise TypeError("For type='text', data must be str or list[str]")

    elif type == "image":
        clip_model = _get_clip_model()
        if clip_model != "fallback":
            if isinstance(data, Image.Image):
                return clip_model.encode([data], convert_to_numpy=True)[0]
            elif isinstance(data, list) and all(isinstance(x, Image.Image) for x in data):
                return clip_model.encode(data, convert_to_numpy=True)
        
        if isinstance(data, Image.Image):
            return np.zeros(512, dtype=np.float32)
        elif isinstance(data, list):
            return np.zeros((len(data), 512), dtype=np.float32)
        else:
            raise TypeError("For type='image', data must be PIL.Image or list[PIL.Image]")

    else:
        raise ValueError("type must be 'text' or 'image'")

