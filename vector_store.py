import os
import json
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, persist_directory: str = "./vector_data"):
        self.persist_directory = persist_directory
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []
        
        os.makedirs(persist_directory, exist_ok=True)
        self.load()
    
    def create_single_embedding(self, text: str) -> List[float]:
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hex_hash = hash_obj.hexdigest()
        
        vector = []
        for i in range(0, min(128, len(hex_hash)), 2):
            byte_pair = hex_hash[i:i+2]
            if len(byte_pair) == 2:
                vector.append(int(byte_pair, 16) / 255.0)
            else:
                vector.append(0.0)
        
        while len(vector) < 128:
            vector.append(0.0)
        
        return vector
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.create_single_embedding(text) for text in texts]
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        try:
            embeddings = self.create_embeddings(documents)
            
            self.documents.extend(documents)
            self.embeddings.extend(embeddings)
            
            if metadata:
                self.metadata.extend(metadata)
            else:
                base_index = len(self.documents) - len(documents)
                self.metadata.extend([{
                    "id": base_index + i,
                    "added": datetime.now().isoformat(),
                    "words": len(doc.split()),
                    "chars": len(doc)
                } for i, doc in enumerate(documents)])
            
            self.save()
            
            print(f"Successfully added {len(documents)} documents")
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))
    
    def search(self, query: str, k: int = 4, min_score: float = 0.1) -> List[Dict]:
        if not self.documents:
            print("No documents in vector store")
            return []
        
        print(f"Searching for: '{query[:50]}...'")
        
        try:
            query_embeddings = self.create_embeddings([query])
            query_embedding = query_embeddings[0]
            
            similarities = []
            for doc_embedding in self.embeddings:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)
            
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                
                if score < min_score:
                    continue
                
                result = {
                    'content': self.documents[idx],
                    'score': float(score),
                    'similarity_percent': f"{score * 100:.1f}%",
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'rank': len(results) + 1
                }
                results.append(result)
            
            print(f"Found {len(results)} relevant results")
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def save(self):
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0',
                'total_documents': len(self.documents)
            }
            
            filepath = os.path.join(self.persist_directory, 'vector_store.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"Saved {len(self.documents)} documents to disk")
            
        except Exception as e:
            print(f"Error saving to disk: {e}")
    
    def load(self):
        filepath = os.path.join(self.persist_directory, 'vector_store.json')
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.documents = data.get('documents', [])
                self.embeddings = data.get('embeddings', [])
                self.metadata = data.get('metadata', [])
                
                print(f"Loaded {len(self.documents)} documents from disk")
                
            except Exception as e:
                print(f"Error loading from disk: {e}")
                self.documents = []
                self.embeddings = []
                self.metadata = []
        else:
            print("No saved vector store found, starting fresh")
    
    def clear(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        filepath = os.path.join(self.persist_directory, 'vector_store.json')
        if os.path.exists(filepath):
            os.remove(filepath)
        
        print("Vector store cleared")
    
    def get_info(self) -> Dict:
        total_chars = sum(len(doc) for doc in self.documents)
        total_words = sum(len(doc.split()) for doc in self.documents)
        
        return {
            "total_documents": len(self.documents),
            "total_words": total_words,
            "total_chars": total_chars,
            "embedding_type": "Simple Hash",
            "embedding_dimensions": len(self.embeddings[0]) if self.embeddings else 0
        }