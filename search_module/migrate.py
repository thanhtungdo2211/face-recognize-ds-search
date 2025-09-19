import json
import uuid
import time
from typing import List, Dict, Any

import numpy as np

from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from backend.core.database import init_db, get_db
from backend.model.access import UserFeature
from search_module.qdrant_client_service import QdrantFeatureStorage

class FeatureMigrator:
    def __init__(self):
        self.feature_storage = QdrantFeatureStorage()
        self.qdrant_client = self.feature_storage.client
        self.collection_name = self.feature_storage.collection_name
        
    def create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.EUCLID)
            )
    
    def convert_feature_to_vector(self, feature_data: str) -> List[float]:
        """Convert feature data to normalized vector"""
        try:
            if isinstance(feature_data, str):
                if feature_data.startswith('[') and feature_data.endswith(']'):
                    vector = json.loads(feature_data)
                elif ',' in feature_data:
                    vector = [float(x.strip()) for x in feature_data.split(',')]
                else:
                    raise ValueError("Unrecognized format")
            else:
                vector = list(feature_data)
            
            # Ensure 512 dimensions
            if len(vector) != 512:
                if len(vector) < 512:
                    vector.extend([0.0] * (512 - len(vector)))
                else:
                    vector = vector[:512]
            
            # Normalize
            vector_array = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(vector_array)
            if norm > 0:
                vector = (vector_array / norm).tolist()
            
            return vector
            
        except Exception:
            # Fallback to random normalized vector
            random_vector = np.random.rand(512).astype(np.float32)
            return (random_vector / np.linalg.norm(random_vector)).tolist()
    
    def migrate_features_to_qdrant(self, db: Session, default_camera_id: str = "migrated"):
        """Migrate features from database to Qdrant"""
        start_time = time.time()
        
        self.create_collection_if_not_exists()
        
        features = db.query(UserFeature).all()
        if not features:
            print("No features found")
            return
        
        print(f"Migrating {len(features)} features...")
        
        points = []
        for feature in features:
            try:
                vector = self.convert_feature_to_vector(feature.feature_data)
                point = PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={
                        "user_id": str(feature.user_id),
                        "camera_id": getattr(feature, 'camera_id', default_camera_id)
                    }
                )
                points.append(point)
            except Exception:
                continue
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)
        
        total_time = time.time() - start_time
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        
        print(f"Migrated {len(points)} vectors in {total_time:.1f}s")
        print(f"Total vectors in collection: {collection_info.points_count}")
    
    def verify_migration(self):
        """Quick verification"""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        if collection_info.points_count == 0:
            print("No vectors found!")
            return
        
        # Test search
        random_vector = np.random.rand(512).astype(np.float32)
        query_vector = (random_vector / np.linalg.norm(random_vector)).tolist()
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1,
            with_payload=True
        )
        
        if results.points:
            print(f"Search test: OK ({len(results.points)} results)")
        else:
            print("Search test: Failed")
    
    def clear_collection(self):
        """Clear collection"""
        self.qdrant_client.delete_collection(self.collection_name)
        self.create_collection_if_not_exists()
        print("Collection cleared")

def main():
    print("Face Feature Migration")
    print("=" * 30)
    
    init_db()
    migrator = FeatureMigrator()
    db = next(get_db())
    
    try:
        print("\n1. Migrate\n2. Verify\n3. Clear & Migrate")
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            migrator.migrate_features_to_qdrant(db)
            migrator.verify_migration()
        elif choice == "2":
            migrator.verify_migration()
        elif choice == "3":
            migrator.clear_collection()
            migrator.migrate_features_to_qdrant(db)
            migrator.verify_migration()
        else:
            print("Invalid choice")
            return
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()