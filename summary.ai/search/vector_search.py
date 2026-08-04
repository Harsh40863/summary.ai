import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class VectorSearchEngine:
    """
    FAISS or numpy-backed vector search engine using Inner Product (cosine similarity
    after L2 normalization) for fast retrieval.
    """

    def __init__(self, dimension):
        self.dimension = dimension
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None
            self.embeddings = None
        self.texts = []

    @property
    def ntotal(self):
        if HAS_FAISS and self.index is not None:
            return self.index.ntotal
        elif self.embeddings is not None:
            return len(self.embeddings)
        return 0

    @staticmethod
    def _normalize(embeddings):
        """L2-normalize embeddings so that inner product equals cosine similarity."""
        emb = np.array(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return emb / norms

    def build_index(self, embeddings, texts):
        """Add embeddings to the search index along with their associated texts."""
        emb = self._normalize(embeddings)
        if HAS_FAISS and self.index is not None:
            self.index.add(emb)
        else:
            self.embeddings = emb
        self.texts = list(texts)

    def search(self, query_embedding, top_k=5):
        """
        Search the index for the top_k most similar vectors.
        """
        q = self._normalize(query_embedding)
        if HAS_FAISS and self.index is not None:
            top_k = min(top_k, self.index.ntotal)
            if top_k <= 0:
                return []
            scores, indices = self.index.search(q, top_k)
            results = []
            for j, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                results.append((int(idx), float(scores[0][j]), self.texts[idx]))
            return results
        else:
            if self.embeddings is None or len(self.embeddings) == 0:
                return []
            top_k = min(top_k, len(self.embeddings))
            if top_k <= 0:
                return []
            scores = np.dot(self.embeddings, q.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in indices:
                results.append((int(idx), float(scores[idx]), self.texts[idx]))
            return results

    @classmethod
    def from_embeddings(cls, embeddings, texts):
        """
        Convenience factory: build a temporary index for one-off searches.
        """
        emb = np.array(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        engine = cls(dimension=emb.shape[1])
        engine.build_index(emb, texts)
        return engine

