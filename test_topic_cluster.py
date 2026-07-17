import unittest
from unittest.mock import patch

import numpy as np

import topic_cluster


class GraphSimilarityTests(unittest.TestCase):
    def test_shared_named_entity_creates_stronger_graph_edge(self):
        titles = [
            "삼성전자 AI 반도체 투자 확대",
            "삼성전자 갤럭시 신제품 공개",
            "현대차 미국 공장 생산 증가",
        ]

        graph = topic_cluster._build_graph_similarity(titles)

        self.assertGreater(graph[0, 1], graph[0, 2])
        self.assertEqual(graph[0, 1], 1.0)
        self.assertEqual(graph[0, 2], 0.0)

    def test_hybrid_similarity_links_graph_related_articles_when_embedding_is_weak(self):
        titles = [
            "삼성전자 AI 반도체 투자 확대",
            "삼성전자 갤럭시 신제품 공개",
            "현대차 미국 공장 생산 증가",
        ]
        # 첫 두 기사의 임베딩 코사인 유사도는 0.30으로 기존 기준(0.55)에 못 미친다.
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.3, np.sqrt(0.91), 0.0],
            [0.0, 0.0, 1.0],
        ])

        hybrid = topic_cluster._build_hybrid_similarity(embeddings, titles)

        self.assertGreaterEqual(hybrid[0, 1], topic_cluster._HYBRID_SIMILARITY_THRESHOLD)
        self.assertLess(hybrid[0, 2], topic_cluster._HYBRID_SIMILARITY_THRESHOLD)

    def test_top_topics_uses_hybrid_graph_and_embedding_clustering(self):
        titles = [
            "삼성전자 AI 반도체 투자 확대",
            "삼성전자 갤럭시 신제품 공개",
            "현대차 미국 공장 생산 증가",
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.3, np.sqrt(0.91), 0.0],
            [0.0, 0.0, 1.0],
        ])

        with patch.object(topic_cluster, "_encode_titles", return_value=(embeddings, 0.45)):
            topics = topic_cluster.get_top_topics(titles, top_n=3)

        samsung_topic = next(topic for topic in topics if "삼성전자" in topic["rep_title"])
        self.assertEqual(samsung_topic["count"], 2)
        self.assertEqual(samsung_topic["match_method"], "임베딩 + 엔티티 그래프")


if __name__ == "__main__":
    unittest.main()
