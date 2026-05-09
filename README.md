# 📰 Korea Finance News Tracker

> 한국 경제 뉴스를 자동 수집하고 산업 섹터별로 분류하는 모니터링 도구
> API 비용 없이 동작 — 로컬 LLM(Ollama)을 활용

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Ollama](https://img.shields.io/badge/Ollama-EXAONE_3.5-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 프로젝트 소개

부트캠프 학습 중 만든 개인 프로젝트입니다. 한국 경제 뉴스를 매일 손으로 찾아보는 비효율을 줄이기 위해, **자동 수집 → 산업 섹터 분류 → LLM 요약**까지 한 번에 처리하는 도구를 만들었습니다.

특히 **API 비용 부담 없이** 로컬 환경에서 LLM을 활용하는 방법을 실험하는 데 초점을 두었습니다.

## ✨ 주요 기능

### 1. 다중 소스 뉴스 수집
- **한국경제 RSS 피드** — 실시간 금융 뉴스
- **네이버 금융 크롤링** — 여러 언론사 통합 뉴스
- 중복 기사 자동 제거

### 2. 산업 섹터 자동 분류 (하이브리드 방식)
- **1단계**: 키워드 사전 매칭 (빠르고 정확)
- **2단계**: LLM 분류 (사전에 없는 신규 키워드 처리)
- 결과 캐싱으로 재실행 시 즉시 로드

지원 섹터: 반도체, 자동차, 이차전지, 바이오/제약, 금융, 게임/엔터, 조선/방산, 철강/소재, 부동산/건설, IT/플랫폼

### 3. 실시간 주가 모니터링
- 네이버 금융에서 관심 종목 시세 수집
- 현재가, 전일 대비, 등락률 카드 형태 표시

### 4. 로컬 LLM 요약
- Ollama + EXAONE 3.5 활용
- 헤드라인 묶음 → 핵심 흐름 3줄 요약

### 5. Streamlit 웹 대시보드
- 사이드바에서 뉴스 소스, 섹터 선택
- 한 화면에서 주가 + 뉴스 + 요약 통합 조회

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| 언어 | Python 3.12 |
| 웹 UI | Streamlit |
| 데이터 수집 | feedparser, requests, BeautifulSoup4 |
| LLM | Ollama (EXAONE 3.5 / Llama 3.2) |
| 설정 | PyYAML |
| 데이터 저장 | JSON, Markdown |

## 📁 프로젝트 구조

```
korea-finance-news-tracker/
├── dashboard.py              # Streamlit 대시보드 (메인)
├── news_tracker.py           # CLI 인터페이스
│
├── feeds/                    # 뉴스 수집
│   ├── hankyung_feed.py      # 한경 RSS
│   └── naver_finance_feed.py # 네이버 금융 크롤링
│
├── classifier/               # 산업 섹터 분류
│   ├── classifier.py         # 하이브리드 메인
│   ├── classifier_keywords.py # 키워드 기반
│   └── classifier_llm.py     # LLM 기반
│
├── stocks/                   # 주가 모니터링
│   └── price_fetcher.py
│
├── config/                   # 설정 파일
│   ├── sectors.yaml          # 섹터별 키워드
│   └── watchlist_stocks.yaml # 관심 종목
│
├── requirements.txt
└── README.md
```

## 🚀 시작하기

### 1. 저장소 복제

```bash
git clone https://github.com/juyoo0729/korea-finance-news-tracker.git
cd korea-finance-news-tracker
```

### 2. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Ollama 설치 및 모델 다운로드

[Ollama 공식 사이트](https://ollama.com/download/windows)에서 설치 후:

```bash
ollama pull exaone3.5:2.4b
```

### 4. 실행

**웹 대시보드 (추천)**

```bash
streamlit run dashboard.py
```

브라우저에서 자동으로 `http://localhost:8501` 열림.

**CLI**

```bash
# 전체 뉴스 수집
python news_tracker.py

# 키워드 필터링
python news_tracker.py --keyword 반도체

# LLM 요약 포함
python news_tracker.py --keyword 반도체 --summarize
```

## 💡 개발 과정에서 배운 점

이 프로젝트의 진짜 가치는 도구 자체보다 **만들면서 부딪힌 문제들을 풀어가는 과정**에 있었습니다.

### 한국어 뉴스의 별명/이형 표기 문제
- 키워드 "삼성전자"로 검색했는데 0건이 나오는 현상
- 실제 헤드라인은 **"삼전"**, **"50만전자"**, **"삼전닉스"** 같은 별명을 더 많이 사용
- 해결: 종목별 별명 그룹을 정의한 사전 구조 도입

### 키워드 사전 vs LLM 분류의 trade-off
- **사전 기반**: 빠르고 정확하지만 신규 키워드 대응 불가
- **LLM 기반**: 유연하지만 느리고 분류가 일관성 떨어짐
- 해결: **하이브리드 분류기** 설계 — 사전으로 80% 처리 후 나머지를 LLM으로

### API 의존성을 피하는 설계
- Claude/GPT API 사용 시 비용과 키 관리 부담
- 해결: **Ollama + EXAONE 로컬 모델**로 전환 — RTX 3070 환경에서 충분히 동작
- 한국어 요약 품질은 EXAONE이 Llama 3.2보다 우수했음

### 데이터 수집의 현실적 제약
- 한국 경제지의 페이월 정책으로 본문 수집 불가
- 해결: 헤드라인 + 요약 + 원문 링크 조합으로 실용적 모니터링 구현

## 🔧 향후 개선 방향

- [ ] DART(전자공시) 연동 — 공시 정보가 뉴스보다 빠르고 정확
- [ ] 더 많은 뉴스 소스 추가 (매일경제, 머니투데이 등)
- [ ] 임베딩 기반 유사 기사 클러스터링
- [ ] 일일 리포트 자동 생성 (이메일/슬랙 알림)
- [ ] 종목별 뉴스 감성 분석

## 📄 라이선스

개인 학습/포트폴리오 용도로 제작됨

## 🙋 만든 사람

[@juyoo0729](https://github.com/juyoo0729)

AIFFEL AI/ML 부트캠프 교육생 (2026.03.11 ~ 2026.09.10)
ML/Data Science 분야 전환 준비 중

---

> ⚠️ 본 도구는 학습 및 개인 모니터링 목적으로 제작되었습니다.
> 투자 자문 도구가 아니며, 투자 판단의 책임은 사용자에게 있습니다.
