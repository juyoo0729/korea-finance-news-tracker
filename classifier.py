import hashlib
import json
from pathlib import Path
from typing import Tuple

from classifier_keywords import classify_by_keywords
from classifier_llm import classify_by_llm

CACHE_PATH = Path(__file__).parent / "classification_cache.json"


class HybridClassifier:
    def __init__(self):
        self._cache: dict = {}
        self._load_cache()

    def _load_cache(self):
        if CACHE_PATH.exists():
            with open(CACHE_PATH, encoding="utf-8") as f:
                self._cache = json.load(f)

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def classify(self, text: str) -> Tuple[str, str]:
        key = self._hash(text)

        if key in self._cache:
            entry = self._cache[key]
            return entry["sector"], entry["method"]

        sector = classify_by_keywords(text)
        if sector is not None:
            method = "키워드"
        else:
            sector = classify_by_llm(text)
            method = "LLM"

        self._cache[key] = {"sector": sector, "method": method}
        return sector, method

    def save_cache(self):
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> dict:
        keyword_count = sum(1 for v in self._cache.values() if v["method"] == "키워드")
        llm_count = sum(1 for v in self._cache.values() if v["method"] == "LLM")
        return {"키워드": keyword_count, "LLM": llm_count}
