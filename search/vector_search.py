import faiss
import numpy as np

class VectorSearchEngine:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def build_index(self, embeddings, texts):
        self.index.add(np.array(embeddings))
        self.texts = texts

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
