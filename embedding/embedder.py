from sentence_transformers import SentenceTransformer
from PIL import Image

# Load models once
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = SentenceTransformer("clip-ViT-B-32")


def get_embeddings(data, type: str = "text"):
    """
    Generate embeddings for text or images.

    Parameters
    ----------
    data : str | list[str] | PIL.Image | list[PIL.Image]
        Text(s) or image(s) to embed.
    type : str
        Either "text" or "image".

    Returns
    -------
    numpy.ndarray
        For a single input -> shape (dim,)
        For a list of inputs -> shape (n, dim)
    """
    if type == "text":
        # Use all-MiniLM-L6-v2 for text (handles longer text better)
        if isinstance(data, str):
            return text_model.encode([data], convert_to_numpy=True)[0]
        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            return text_model.encode(data, convert_to_numpy=True)
        else:
            raise TypeError("For type='text', data must be str or list[str]")

    elif type == "image":
        # Use CLIP for images
        if isinstance(data, Image.Image):
            return clip_model.encode([data], convert_to_numpy=True)[0]
        elif isinstance(data, list) and all(isinstance(x, Image.Image) for x in data):
            return clip_model.encode(data, convert_to_numpy=True)
        else:
            raise TypeError("For type='image', data must be PIL.Image or list[PIL.Image]")

    else:
        raise ValueError("type must be 'text' or 'image'")
