import numpy as np
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer

class VectorSimilaritySearch:
    def __init__(self, dimension: int = 384):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(dimension)
        self.text_lookup: Dict[int, str] = {}
        self.current_id = 0
    
    def add_texts(self, texts: List[str]) -> None:
        # Convert texts to vectors
        vectors = self.model.encode(texts)
        
        # Add to FAISS index
        for i, text in enumerate(texts):
            vector = vectors[i].reshape(1, -1)
            self.index.add(vector.astype('float32'))
            self.text_lookup[self.current_id] = text
            self.current_id += 1
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        # Convert query to vector
        query_vector = self.model.encode([query])[0].reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(
            query_vector.astype('float32'), k
        )
        
        # Format results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                similarity = 1 / (1 + dist)  # Convert distance to similarity
                results.append((self.text_lookup[idx], similarity))
        
        return results

    def batch_search(
        self, 
        queries: List[str], 
        k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        # Convert queries to vectors
        query_vectors = self.model.encode(queries)
        
        # Search index
        distances, indices = self.index.search(
            query_vectors.astype('float32'), k
        )
        
        # Format results
        all_results = []
        for query_indices, query_distances in zip(indices, distances):
            results = []
            for idx, dist in zip(query_indices, query_distances):
                if idx != -1:
                    similarity = 1 / (1 + dist)
                    results.append((self.text_lookup[idx], similarity))
            all_results.append(results)
            
        return all_results


searcher = VectorSimilaritySearch()

# Add documents
docs = [
    "Neural networks are powerful",
    "Machine learning transforms industries",
    "Deep learning revolutionizes AI",
    "Computer vision analyzes images"
]
searcher.add_texts(docs)

# Single search
results = searcher.search("AI and neural networks")
print(f"Top matches: {results}")

# Batch search
queries = ["machine learning", "computer vision"]
batch_results = searcher.batch_search(queries)
print(f"Batch results: {batch_results}")