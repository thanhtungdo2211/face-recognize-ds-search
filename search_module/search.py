import uuid
import time
from typing import List, Union, Any, Dict

from qdrant_client.models import PointStruct #type: ignore

from search_module.qdrant_client_service import QdrantFeatureStorage

feature_storage = QdrantFeatureStorage()

def upsert(
    user_id: str,
    features: Union[List[float], List[List[float]]],
    camera_id: str,
) -> Dict[str, Any]:
    try:    
        # Prepare points for upsert
        points = []
        for feature_vector in features:
            # Ensure vector is 512-dimensional
            if len(feature_vector) != 512:
                raise ValueError(f"Feature vector must be 512-dimensional, got {len(feature_vector)}")
            
            payload = {
                "user_id": user_id,
                "camera_id": camera_id
            }
            
            feature_id = uuid.uuid4().hex
  
            point = PointStruct(
                id=feature_id,
                vector=feature_vector,
                payload=payload
            )
            points.append(point)
        
        # Perform upsert
        operation_info = feature_storage.client.upsert(
            collection_name=feature_storage.collection_name,
            points=points
        )
        
        return {
            "status": "success",
            "operation_id": operation_info.operation_id if hasattr(operation_info, 'operation_id') else None,
            "upserted_count": len(points),
            "user_id": user_id
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "upserted_count": 0
        }

def search(
    query_vector: List[float],  
    limit: int = 100,
    similarity_threshold: float = 1.15
) -> Dict[str, Any]:
    try:
        if len(query_vector) != 512:
            raise ValueError(f"Query vector must be 512-dimensional, got {len(query_vector)}")

        # Perform search
        # start = time.time()
        search_results = feature_storage.client.query_points(
            collection_name=feature_storage.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
        # end = time.time()
        # print(f"Search time: {(end-start)*1000:.2f}ms")
        
        # Check if we have any results
        if not search_results:
            return {
                "result": False,
                "score": float('inf'),
                "user_id": None
            }
        
        # Find the best match (lowest score for Euclidean distance)
        best_match = search_results[0]  # Results are already sorted by score
        best_score = best_match.score
        best_user_id = best_match.payload.get('user_id')
        
        # Check if the best match meets the similarity threshold
        is_similar = best_score <= similarity_threshold
        
        return {
            "result": is_similar,
            "score": best_score,
            "user_id": best_user_id if is_similar else None
        }
        
    except Exception as e:
        print(f"Search error: {e}")
        return {
            "result": False,
            "score": float('inf'),
            "user_id": None
        }
    