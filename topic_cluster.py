"""
topic_cluster.py — 뉴스 헤드라인 하이브리드 주제 클러스터링

sentence-transformers 임베딩 유사도와 헤드라인의 엔티티 연결 그래프를
결합하고, AgglomerativeClustering으로 관련 뉴스를 묶어
가장 많이 등장한 주제 TOP N을 반환한다.
"""

from __future__ import annotations

import re

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SK_AVAILABLE = True
except ImportError:
    _SK_AVAILABLE = False

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 코사인 거리 임계값 (0~2 범위). 낮을수록 같은 클러스터로 묶는 기준이 엄격해짐.
# 0.45 ≈ 코사인 유사도 0.55 이상인 제목끼리 같은 주제로 분류.
_DISTANCE_THRESHOLD = 0.45

# 의미 유사도만으로 놓치는 동일 기업·기관 기사를 그래프 연결로 보강한다.
_EMBEDDING_WEIGHT = 0.60
_GRAPH_WEIGHT = 0.40
_HYBRID_SIMILARITY_THRESHOLD = 0.55

_GRAPH_STOPWORDS = {
    "관련", "뉴스", "오늘", "올해", "작년", "지난해", "기자", "단독",
    "투자", "확대", "증가", "감소", "급등", "급락", "상승", "하락",
    "공개", "발표", "전망", "시장", "실적", "주가", "경제", "금융",
    "대한", "위한", "통해", "한다", "했다", "된다", "나선다",
}
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&.+-]*")

_model: "SentenceTransformer | None" = None


def get_topic_cluster_status() -> dict[str, bool]:
    """주제 클러스터링에 필요한 선택 의존성 설치 상태를 반환한다."""
    return {
        "sentence_transformers": _ST_AVAILABLE,
        "scikit_learn": _SK_AVAILABLE,
    }


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
    return _model


def _encode_titles(clean: list[str]) -> tuple[np.ndarray, float] | None:
    if _ST_AVAILABLE:
        try:
            model = _get_model()
            embeddings = model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
            return embeddings, _DISTANCE_THRESHOLD
        except Exception as e:
            print(f"[topic_cluster] sentence-transformers 모델 로드 실패, TF-IDF로 대체: {e}")

    if not _SK_AVAILABLE:
        return None

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    return vectorizer.fit_transform(clean).toarray(), 0.75


def _extract_graph_nodes(title: str) -> tuple[set[str], set[str]]:
    """제목에서 그래프 노드를 추출한다.

    3자 이상의 고유명사 후보(기업·기관·상품 등)는 강한 노드로 보고,
    2자 토큰은 보조 노드로 사용한다. 범용 뉴스 표현은 제외한다.
    """
    tokens = {
        token.casefold()
        for token in _TOKEN_RE.findall(title)
        if len(token) >= 2 and token.casefold() not in _GRAPH_STOPWORDS
    }
    strong = {
        token for token in tokens
        if len(token) >= 3 or any(ch.isdigit() for ch in token)
    }
    return strong, tokens - strong


def _build_graph_similarity(titles: list[str]) -> np.ndarray:
    """기사-엔티티 이분 그래프의 공유 노드를 기사 간 유사도로 변환한다."""
    size = len(titles)
    graph = np.eye(size, dtype=float)
    nodes = [_extract_graph_nodes(title) for title in titles]

    for left in range(size):
        left_strong, left_weak = nodes[left]
        for right in range(left + 1, size):
            right_strong, right_weak = nodes[right]
            if left_strong & right_strong:
                similarity = 1.0
            else:
                shared_weak = left_weak & right_weak
                weak_base = min(len(left_weak), len(right_weak))
                similarity = 0.5 * len(shared_weak) / weak_base if weak_base else 0.0
            graph[left, right] = graph[right, left] = similarity

    return graph


def _build_hybrid_similarity(embeddings: np.ndarray, titles: list[str]) -> np.ndarray:
    """정규화 임베딩 유사도와 엔티티 그래프 유사도를 결합한다."""
    vectors = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    semantic = np.clip(normalized @ normalized.T, 0.0, 1.0)
    graph = _build_graph_similarity(titles)
    hybrid = _EMBEDDING_WEIGHT * semantic + _GRAPH_WEIGHT * graph
    np.fill_diagonal(hybrid, 1.0)
    return hybrid


def get_top_topics(titles: list[str], top_n: int = 3) -> list[dict]:
    """헤드라인을 임베딩+엔티티 그래프로 묶어 상위 top_n 주제를 반환한다.

    Parameters
    ----------
    titles : 뉴스 제목 문자열 리스트
    top_n  : 반환할 주제 수

    Returns
    -------
    list[dict]  기사 수 내림차순
        rank      : 순위 (1부터)
        rep_title : 클러스터 대표 헤드라인 (클러스터 중심에 가장 가까운 제목)
        count     : 묶인 기사 수
        headlines : 해당 클러스터 전체 제목 리스트
    """
    if not _SK_AVAILABLE:
        return []

    clean = [t.strip() for t in titles if t.strip()]
    if len(clean) < 2:
        return []

    try:
        encoded = _encode_titles(clean)
        if encoded is None:
            return []
        embeddings, distance_threshold = encoded

        hybrid_similarity = _build_hybrid_similarity(embeddings, clean)
        hybrid_distance = 1.0 - hybrid_similarity

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - _HYBRID_SIMILARITY_THRESHOLD,
        )
        labels = clustering.fit_predict(hybrid_distance)

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)

        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)[:top_n]

        result = []
        for rank, indices in enumerate(sorted_clusters, 1):
            cluster_embs = embeddings[indices]
            centroid = cluster_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            sims = cluster_embs @ centroid
            rep_idx = indices[int(np.argmax(sims))]

            result.append({
                "rank": rank,
                "rep_title": clean[rep_idx],
                "count": len(indices),
                "headlines": [clean[i] for i in indices],
                "match_method": "임베딩 + 엔티티 그래프",
            })

        return result

    except Exception as e:
        print(f"[topic_cluster] 클러스터링 실패 (모델 로드 또는 네트워크 오류): {e}")
        return []
