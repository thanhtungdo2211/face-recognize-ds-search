import os 
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv("./.env")
# Qdrant configuration
QDRANT_HOST = os.getenv('QDRANT_HOST', '192.168.6.194')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6399'))
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'user_features')

class QdrantFeatureStorage:
    def __init__(self):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self.collection_name = COLLECTION_NAME
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't"""
        try:
            existing_collections = self.client.get_collections()
            collection_names = [col.name for col in existing_collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=512,
                        distance=Distance.EUCLID  # Better for face embeddings
                        # distance=Distance.COSINE  # Better for face embeddings
                    )
                )
                print(f"Collection '{self.collection_name}' created")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise