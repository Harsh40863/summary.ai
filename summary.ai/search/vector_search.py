import faiss
import numpy as np

class VectorSearchEngine:
    """
    FAISS-backed vector search engine using Inner Product (cosine similarity
    after L2 normalization) for fast approximate nearest-neighbour retrieval.
    """

    def __init__(self, dimension):
        # Inner Product index: after L2-normalising vectors, IP == cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []

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
        """Add embeddings to the FAISS index along with their associated texts."""
        emb = self._normalize(embeddings)
        self.index.add(emb)
        self.texts = list(texts)

    def search(self, query_embedding, top_k=5):
        """
        Search the index for the top_k most similar vectors.

        Parameters
        ----------
        query_embedding : array-like
            1-D or 2-D query vector(s).
        top_k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list[tuple[int, float, str]]
            Each tuple is (index, cosine_similarity_score, associated_text).
        """
        q = self._normalize(query_embedding)
        top_k = min(top_k, self.index.ntotal)
        if top_k <= 0:
            return []
        scores, indices = self.index.search(q, top_k)
        results = []
        for j, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            results.append((int(idx), float(scores[0][j]), self.texts[idx]))
        return results

    @classmethod
    def from_embeddings(cls, embeddings, texts):
        """
        Convenience factory: build a temporary index for one-off searches
        (e.g. chunk-level search within a single document).
        """
        emb = np.array(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        engine = cls(dimension=emb.shape[1])
        engine.build_index(emb, texts)
        return engine
