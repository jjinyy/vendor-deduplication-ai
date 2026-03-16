"""
ANN(Approximate Nearest Neighbor) 인덱스 추상화 레이어
FAISS HNSW > hnswlib > sklearn 순으로 자동 선택
"""
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# 라이브러리 가용성 확인
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ANNIndex:
    """ANN 인덱스 추상화 클래스"""
    
    def __init__(self, dim: int, method: str = 'auto'):
        """
        Args:
            dim: 벡터 차원
            method: 'auto', 'faiss', 'hnswlib', 'sklearn'
        """
        self.dim = dim
        self.method = method
        self.index = None
        self.index_type = None
        self._ids = None  # 인덱스에 저장된 row id 리스트
        self._n_samples = 0  # 인덱스에 저장된 샘플 수
        
        # 사용 가능한 방법 선택
        if method == 'auto':
            if FAISS_AVAILABLE:
                self.index_type = 'faiss'
            elif HNSWLIB_AVAILABLE:
                self.index_type = 'hnswlib'
            elif SKLEARN_AVAILABLE:
                self.index_type = 'sklearn'
            else:
                raise RuntimeError("No ANN library available. Install faiss-cpu, hnswlib, or scikit-learn")
        else:
            self.index_type = method
        
        logger.info(f"ANN Index 초기화: method={self.index_type}, dim={dim}")
    
    def build(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """
        인덱스 구축
        
        Args:
            vectors: (N, dim) 형태의 정규화된 벡터 배열
            ids: 각 벡터에 대응하는 row id 리스트 (None이면 0..N-1)
        """
        N = vectors.shape[0]
        if ids is None:
            ids = list(range(N))
        self._ids = ids
        self._n_samples = N
        
        if self.index_type == 'faiss':
            self._build_faiss(vectors)
        elif self.index_type == 'hnswlib':
            self._build_hnswlib(vectors)
        elif self.index_type == 'sklearn':
            self._build_sklearn(vectors)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"ANN Index 구축 완료: {N}개 벡터")
    
    def _build_faiss(self, vectors: np.ndarray):
        """FAISS HNSW 인덱스 구축"""
        N, dim = vectors.shape
        
        # HNSW 파라미터
        M = 32  # 각 노드의 최대 연결 수
        ef_construction = 200  # 구축 시 탐색 범위
        
        # L2 거리 기반 인덱스 (cosine similarity는 normalize 후 L2와 동일)
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        
        # 벡터 추가
        self.index.add(vectors.astype('float32'))
    
    def _build_hnswlib(self, vectors: np.ndarray):
        """hnswlib 인덱스 구축"""
        N, dim = vectors.shape
        
        # HNSW 파라미터
        M = 32
        ef_construction = 200
        max_elements = N
        
        self.index = hnswlib.Index(space='l2', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.add_items(vectors.astype('float32'), ids=list(range(N)))
    
    def _build_sklearn(self, vectors: np.ndarray):
        """sklearn NearestNeighbors 인덱스 구축 (fallback)"""
        # sklearn은 메모리에 모든 벡터 저장
        self.index = NearestNeighbors(n_neighbors=min(200, vectors.shape[0]), 
                                      metric='cosine', algorithm='brute')
        self.index.fit(vectors)
        self._vectors = vectors  # 검색을 위해 벡터 저장
    
    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        검색 수행
        
        Args:
            query_vectors: (M, dim) 형태의 쿼리 벡터 배열
            top_k: 반환할 최근접 이웃 수
        
        Returns:
            (indices, distances): 
            - indices: (M, top_k) 형태의 이웃 인덱스 (row id)
            - distances: (M, top_k) 형태의 거리 (낮을수록 유사)
        """
        # top_k를 실제 샘플 수에 맞게 조정
        actual_top_k = min(top_k, self._n_samples)
        if actual_top_k < 1:
            # 샘플이 없으면 빈 결과 반환
            M = query_vectors.shape[0] if len(query_vectors.shape) > 1 else 1
            return np.array([[]] * M), np.array([[]] * M)
        
        if self.index_type == 'faiss':
            return self._search_faiss(query_vectors, actual_top_k)
        elif self.index_type == 'hnswlib':
            return self._search_hnswlib(query_vectors, actual_top_k)
        elif self.index_type == 'sklearn':
            return self._search_sklearn(query_vectors, actual_top_k)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _search_faiss(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS 검색"""
        ef_search = max(top_k * 2, 50)  # 검색 시 탐색 범위
        self.index.hnsw.efSearch = ef_search
        
        distances, indices = self.index.search(query_vectors.astype('float32'), top_k)
        
        # 인덱스를 row id로 변환
        row_ids = np.array([[self._ids[idx] for idx in row] for row in indices])
        
        return row_ids, distances
    
    def _search_hnswlib(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """hnswlib 검색"""
        ef_search = max(top_k * 2, 50)
        self.index.set_ef(ef_search)
        
        all_indices = []
        all_distances = []
        
        for qv in query_vectors:
            indices, distances = self.index.knn_query(qv.astype('float32'), k=top_k)
            all_indices.append([self._ids[idx] for idx in indices[0]])
            all_distances.append(distances[0])
        
        return np.array(all_indices), np.array(all_distances)
    
    def _search_sklearn(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """sklearn 검색 (fallback)"""
        # top_k를 실제 샘플 수에 맞게 한 번 더 확인 (안전장치)
        actual_top_k = min(top_k, self._n_samples)
        if actual_top_k < 1:
            M = query_vectors.shape[0] if len(query_vectors.shape) > 1 else 1
            return np.array([[]] * M), np.array([[]] * M)
        
        distances, indices = self.index.kneighbors(query_vectors, n_neighbors=actual_top_k)
        
        # 인덱스를 row id로 변환
        row_ids = np.array([[self._ids[idx] for idx in row] for row in indices])
        
        return row_ids, distances


def build_bucket_indices(row_meta: Dict, 
                        embeddings: np.ndarray,
                        bucket_key_fn,
                        row_id_to_pos: Optional[Dict] = None) -> Dict[Tuple, ANNIndex]:
    """
    Bucket별로 ANN 인덱스 구축
    
    Args:
        row_meta: {row_id: {country_key, script_key, ...}}
        embeddings: (N, dim) 형태의 임베딩 배열 (순서대로 정렬되어 있어야 함)
        bucket_key_fn: (row_id, row_meta) -> bucket_key 함수
        row_id_to_pos: {row_id: position_in_embeddings} 매핑 (None이면 순서대로 가정)
    
    Returns:
        {bucket_key: ANNIndex}
    """
    # Bucket별로 row id 그룹화
    bucket_ids = {}
    for row_id, meta in row_meta.items():
        bucket_key = bucket_key_fn(row_id, meta)
        if bucket_key not in bucket_ids:
            bucket_ids[bucket_key] = []
        bucket_ids[bucket_key].append(row_id)
    
    # Bucket별 인덱스 구축
    bucket_indices = {}
    dim = embeddings.shape[1]
    
    for bucket_key, ids in bucket_ids.items():
        if len(ids) < 2:
            continue  # 후보가 1개 이하면 인덱스 불필요
        
        # 해당 bucket의 벡터 추출
        if row_id_to_pos:
            # row_id를 position으로 변환
            positions = [row_id_to_pos[row_id] for row_id in ids]
            bucket_vectors = embeddings[positions]
        else:
            # row_id가 이미 정수형 인덱스라고 가정
            bucket_vectors = embeddings[ids]
        
        # 인덱스 구축
        index = ANNIndex(dim=dim)
        index.build(bucket_vectors, ids=ids)
        bucket_indices[bucket_key] = index
        
        logger.info(f"Bucket {bucket_key}: {len(ids)}개 벡터 인덱스 구축 완료")
    
    return bucket_indices
