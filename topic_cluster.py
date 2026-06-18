"""
topic_cluster.py — 뉴스 헤드라인 임베딩 기반 주제 클러스터링

sentence-transformers로 헤드라인을 벡터화하고,
AgglomerativeClustering으로 유사 제목을 묶어
가장 많이 등장한 주제 TOP N을 반환한다.
"""

from __future__ import annotations

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


def get_top_topics(titles: list[str], top_n: int = 3) -> list[dict]:
    """헤드라인 리스트를 받아 임베딩 → 클러스터링 → 상위 top_n 주제를 반환한다.

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

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(embeddings)

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
            })

        return result

    except Exception as e:
        print(f"[topic_cluster] 클러스터링 실패 (모델 로드 또는 네트워크 오류): {e}")
        return []
