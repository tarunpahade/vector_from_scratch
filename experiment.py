import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import json
import math

class BasicVectorSearch:
    def __init__(self):
        self.documents = []
        self.doc_vectors = []
        self.sources = []
        
    def _create_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        words = set()
        for text in texts:
            words.update(text.lower().split())
        return {word: idx for idx, word in enumerate(words)}
        
    def _text_to_vector(self, text: str, vocab: Dict[str, int]) -> np.ndarray:
        vector = np.zeros(len(vocab))
        words = Counter(text.lower().split())
        for word, count in words.items():
            if word in vocab:
                vector[vocab[word]] = count
        return vector / (np.linalg.norm(vector) + 1e-8)  # Normalize
        
    def add_documents(self, documents: List[Dict]):
        """
        Add documents in format:
        {"pdfName": "name", "content": "text", "source": "source"}
        """
        # Input validation
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            # Extract texts from documents
            texts = [doc["content"] for doc in documents]
            vocab = self._create_vocabulary(texts)
            
            # Process documents
            for doc in documents:
                if not all(key in doc for key in ["pdfName", "content", "source"]):
                    raise KeyError(f"Missing required keys in document: {doc}")
                    
                self.documents.append(doc)
                vector = self._text_to_vector(doc["content"], vocab)
                self.doc_vectors.append(vector)
                self.sources.append(doc["source"])
                
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            raise
            
    def search(self, query: str, k: int = 3) -> List[Dict]:
        vocab = self._create_vocabulary([query] + [doc["content"] for doc in self.documents])
        query_vector = self._text_to_vector(query, vocab)
        
        similarities = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            # Cosine similarity
            similarity = np.dot(query_vector, doc_vector)
            similarities.append((similarity, idx))
            
        # Sort by similarity
        results = []
        for sim, idx in sorted(similarities, reverse=True)[:k]:
            results.append({
                "document": self.documents[idx],
                "similarity": float(sim),
                "source": self.sources[idx]
            })
            
        return results

# Example usage:
if __name__ == "__main__":
    # Sample documents
    documents = [
        {
            "pdfName": "Augrade1",
            "content": "Machine learning is transforming industries",
            "source": "tech_report_2023.pdf"
        },
        {
            "pdfName": "Augrade2",
            "content": "AI systems are becoming more powerful",
            "source": "ai_review.pdf"
        },
        {
            "pdfName": "Augrade3",
            "content": "Deep learning revolutionizes computer vision",
            "source": "vision_paper.pdf"
        }
    ]
    
    # Initialize and add documents
    searcher = BasicVectorSearch()
    searcher.add_documents(documents)
    
    # Search
    results = searcher.search("machine learning AI")
    print(json.dumps(results, indent=2))