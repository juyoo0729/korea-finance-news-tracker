# 📰 Korea Finance News Tracker

> 한국 주식·코인·경제 뉴스를 한 화면에서 확인하고, 기술적 조건에 맞는 **단기 관찰 종목**을 좁혀보는 Streamlit 대시보드입니다.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 프로젝트 원칙

이 프로젝트는 시장 데이터를 정리하고 관찰 조건을 시각화하는 학습용 도구입니다.

- 후보 점수는 매수·매도 추천이나 수익 예측이 아닙니다.
- 최종 점수와 기술지표 계산은 Python의 결정적 규칙으로 수행합니다.
- LLM은 뉴스 분류와 요약을 보조하며 투자 판단을 내리지 않습니다.
- 백테스트 결과는 전략 유효성의 증명이 아닙니다.
- 토스증권·빗썸 API를 설정하면 **실제 주문 전송 기능이 활성화될 수 있습니다.** 주문 내용과 계좌를 반드시 직접 확인해야 합니다.

## 주요 기능

### 1. 시장·자산 통합 화면

- FinanceDataReader 기반 KOSPI·KOSDAQ 전종목 스냅샷
- 등락률, 거래대금, 시가총액 상위 종목
- 실제 시장 인덱스의 최신 거래일과 스냅샷 조회 시각 표시
- 토스증권 주식 자산·예수금·보유 종목 조회
- 빗썸 KRW 자산·보유 코인 조회
- 빗썸 KRW 마켓 시세, 등락률, 24시간 거래대금 표시

### 2. 단기 관찰 종목 TOP 10

전종목을 가볍게 거른 뒤 일부 후보만 일봉으로 보강하는 2단계 구조입니다.

**1단계 — 시장 스냅샷**

- 거래대금·시가총액 필터
- 등락률과 유동성 기반 점수
- 급등 과열과 큰 당일 하락 위험 반영
- 동점 시 거래대금 → 시가총액 → 티커 순으로 결정
- 상위 약 30개를 2단계로 전달

**2단계 — 일봉 기술지표**

- 최근 일봉 거래량
- MA20
- RSI(14)
- 5일 이동평균의 1차·2차 미분
- 20일 모멘텀
- 추세 지속, 과매도 반전, 거래량 이상 등 관찰 유형 표시

장중 미완성 최신봉은 모든 기술지표에서 제외합니다. 일봉 조회 실패나 데이터 부족을 숨기지 않고 후보별 상태·오류·분석 기준일을 표시합니다.

시장 데이터는 5분 캐시하며, 동일 날짜라도 가격·거래대금 등 스냅샷이 바뀌면 관찰 종목을 다시 분석합니다.

### 3. 종목 상세 관찰

관찰 종목을 선택하면 다음 정보를 확인할 수 있습니다.

- 점수 근거와 주의사항
- 일봉 분석 상태와 기준일
- 토스증권 실시간 가격과 FinanceDataReader 지연 시세 차이
- RSI, MA5·MA20, 가속도 변곡점
- 상방 조건 우세·하방 위험 우세·중립 상태
- 최근 조건 발생 이벤트와 관찰 기준선
- TradingView Webhook 최신 신호

모든 표시는 기술적 조건을 분류한 것이며 매매 지시가 아닙니다.

### 4. 뉴스 수집·분류·요약

- 한경 금융 RSS
- 네이버 금융 전체 뉴스
- 네이버 금융 종목별 뉴스
- URL 기준 중복 제거
- 키워드 우선 + 미분류 기사만 Claude로 보완하는 하이브리드 섹터 분류
- Claude Haiku 기반 헤드라인 3줄 요약
- 외부 제목·언론사·요약문 HTML 이스케이프
- 뉴스 링크는 절대 HTTP(S) URL만 허용

### 5. 인기 주제 TOP 3 — 임베딩 + 엔티티 그래프

- 다국어 sentence-transformers 임베딩 의미 유사도: **60%**
- 기업·기관 등 공유 엔티티 그래프 유사도: **40%**
- 결합 거리 기반 AgglomerativeClustering
- 클러스터 중심에 가까운 헤드라인을 대표 제목으로 선택

별도 그래프 데이터베이스를 사용하는 대규모 Graph RAG가 아니라, 헤드라인 엔티티 연결로 임베딩 검색을 보강하는 **경량 하이브리드 그래프 방식**입니다.

### 6. 검색 및 주문

주식·코인을 이름이나 코드로 검색하고 지정가 주문을 전송할 수 있습니다.

- 주식: 토스증권 Open API
- 코인: 빗썸 API
- 토스 주문 계좌를 화면에서 명시적으로 선택
- 주문 내용별 잠금 해제와 확인 체크
- 주문 내용이 바뀌면 확인 상태 초기화
- 빗썸 최소 주문금액 5,000원 검사
- 같은 날짜·계좌·주문 내용에는 동일한 클라이언트 주문 ID를 전송하고 세션 내 재전송 차단

> 이 기능은 모의 주문이 아닙니다. API 키에 거래 권한이 있으면 실제 주문이 접수될 수 있습니다. 처음에는 조회 전용 권한 또는 별도 테스트 계좌 사용을 권장합니다.

네트워크 타임아웃은 주문 실패를 의미하지 않습니다. 결과를 확인할 수 없을 때는 재전송하지 말고 증권사·거래소 앱에서 접수·체결 여부를 먼저 확인하세요. 클라이언트 주문 ID는 중복 위험을 줄이기 위한 보조 수단이며, 실제 멱등 처리 범위는 각 API 정책을 따릅니다.

### 7. 백테스트

- 과거 거래일의 후보 TOP 10 재현
- +5·+10·+20거래일 수익률과 KOSPI 비교
- 승률, 평균 초과수익, 점수 구간별 결과
- 기준일 이후 데이터가 점수에 섞이지 않도록 provider 주입 방식 사용
- 일봉 데이터는 `data_cache/`, 결과는 `results/`에 저장

현재 포함된 소표본 결과는 기준일이 3개뿐이고 KOSPI 대비 성과도 음수였습니다. 따라서 관찰 점수를 투자 추천으로 해석할 근거가 없습니다. 더 긴 기간, 거래비용, 정확한 과거 유니버스를 포함한 검증이 필요합니다.

## 기술 스택

| 분류 | 기술 |
|---|---|
| 언어 | Python 3.11+ |
| 웹 UI | Streamlit |
| 시장 데이터 | FinanceDataReader |
| 주식 연동 | 토스증권 Open API |
| 코인 연동 | 빗썸 API |
| 뉴스 수집 | feedparser, requests, BeautifulSoup4 |
| 데이터 처리 | pandas, NumPy |
| 차트 | Plotly |
| 뉴스 임베딩 | sentence-transformers |
| 클러스터링 | scikit-learn |
| LLM | Anthropic Claude Haiku |
| 설정 | python-dotenv, PyYAML, Streamlit secrets |

## 프로젝트 구조

```text
korea-finance-news-tracker/
├── dashboard.py              # Streamlit 메인 대시보드
├── market_data.py            # 시장 스냅샷·거래일·캐시 키
├── scorer.py                 # 단기 관찰 종목 2단계 점수
├── trade_signal.py           # 기술적 상태·조건 이벤트·차트
├── backtest.py               # 과거 후보 재현과 벤치마크 비교
├── topic_cluster.py          # 임베딩 + 엔티티 그래프 주제 묶기
├── web_safety.py             # 외부 HTML 문자열·URL 검증
├── toss_client.py            # 토스 시세·자산·주문
├── bithumb_client.py         # 빗썸 시세·자산·주문
├── tradingview_signals.py    # TradingView Webhook 수신·신호 조회
├── llm_client.py             # Anthropic Claude 공통 클라이언트
├── classifier.py             # 하이브리드 섹터 분류
├── classifier_keywords.py    # 키워드 분류
├── classifier_llm.py         # LLM 분류
├── hankyung_feed.py          # 한경 RSS
├── naver_finance_feed.py     # 네이버 금융 뉴스
├── news_tracker.py           # 뉴스 전용 CLI
├── test_scorer.py
├── test_market_data.py
├── test_order_safety.py
├── test_topic_cluster.py
├── test_web_safety.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 설치

### Windows

```bash
git clone https://github.com/juyoo0729/korea-finance-news-tracker.git
cd korea-finance-news-tracker
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS·Linux

```bash
git clone https://github.com/juyoo0729/korea-finance-news-tracker.git
cd korea-finance-news-tracker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 설정

### 1. 환경변수

프로젝트 루트의 `.env`에 필요한 항목만 설정합니다.

```dotenv
# 뉴스 분류·요약
ANTHROPIC_API_KEY=your-anthropic-api-key

# 토스증권 시세·자산·주문
TOSS_CLIENT_ID=your-client-id
TOSS_CLIENT_SECRET=your-client-secret

# 빗썸 자산·주문
BITHUMB_ACCESS_KEY=your-access-key
BITHUMB_SECRET_KEY=your-secret-key

# TradingView Webhook
TRADINGVIEW_WEBHOOK_SECRET=replace-with-a-random-secret
TRADINGVIEW_SIGNAL_VALID_MINUTES=1440
TRADINGVIEW_WEBHOOK_PORT=8787
```

API 키를 Git에 커밋하지 마세요. `.env`는 `.gitignore`에 포함되어 있습니다.

- Anthropic 키가 없으면 LLM 요약·보완 분류를 사용할 수 없습니다.
- 토스 키가 없으면 토스 시세·자산·주문 기능이 비활성화됩니다.
- 빗썸 개인 키가 없어도 공개 코인 시세는 조회할 수 있지만 자산·주문은 사용할 수 없습니다.
- 거래 API 키는 필요한 최소 권한만 부여하세요.

### 2. 대시보드 비밀번호

`.streamlit/secrets.toml`을 만들고 강한 비밀번호를 설정합니다.

```toml
password = "replace-with-a-strong-password"
```

이 파일도 `.gitignore`에 포함되어 있습니다. 현재 로그인은 단일 공용 비밀번호 방식이므로 인터넷에 공개 배포할 때는 별도 인증·접근제어를 추가해야 합니다.

## 실행

### Windows 실행 스크립트

```bash
run_dashboard.cmd
```

### 직접 실행

```bash
.venv\Scripts\python.exe -m streamlit run dashboard.py
```

macOS·Linux:

```bash
python -m streamlit run dashboard.py
```

기본 접속 주소는 `http://localhost:8504`입니다.

TradingView Webhook:

```text
http://localhost:8787/tradingview-webhook
```

지원 신호는 `BUY`, `SELL`, `NEUTRAL`이며 기존 `UP`, `DOWN`은 각각 `BUY`, `SELL`로 정규화합니다.

## CLI

뉴스 수집·요약:

```bash
python news_tracker.py --keyword 반도체 --summarize
python news_tracker.py --keyword SK하이닉스 --summarize --model claude-haiku-4-5-20251001
```

백테스트:

```bash
python backtest.py --start 2026-01-01 --end 2026-05-31 --interval 5
python backtest.py --start 2026-05-10 --end 2026-06-10 --interval 10 --universe 300
```

## 테스트

```bash
python -m unittest discover -v
```

현재 회귀 테스트는 다음을 확인합니다.

- 횡보 RSI 중립 처리와 급락 위험
- 점수 우선 순위와 유동성 동점 처리
- 일봉 분석 실패 상태와 장중 미완성봉 제외
- 실제 시장 거래일과 스냅샷 키
- 임베딩 + 엔티티 그래프 결합
- HTML 이스케이프와 위험 URL 스킴 차단

## 캐시와 데이터 기준

| 데이터 | 캐시 |
|---|---:|
| 시장 스냅샷 | 5분 |
| 뉴스·주제 | 5분 |
| 토스 실시간 가격 | 15초 |
| TradingView 최신 신호 | 15초 |
| 빗썸 시세·자산 | 30초 |

화면에는 시장 기준 거래일, 스냅샷 조회 시각, 일봉 분석 기준일을 구분해 표시합니다. 서로 다른 API와 캐시를 사용하므로 모든 값이 완전히 같은 시점의 데이터라고 가정하면 안 됩니다.

## 알려진 한계

- 무료 데이터만으로 과거 시점의 정확한 전종목 거래대금·시가총액 유니버스를 재현하기 어렵습니다.
- 백테스트에는 생존 편향과 근사 시가총액이 포함될 수 있습니다.
- LLM 요약은 원문을 잘못 해석할 수 있으므로 반드시 기사 원문을 확인해야 합니다.
- 뉴스 그래프는 헤드라인 엔티티 기반 경량 연결이며 완전한 지식그래프가 아닙니다.
- 단일 공용 비밀번호는 다중 사용자 서비스용 인증 체계가 아닙니다.
- 주문 API의 응답과 실제 체결 상태는 거래소·증권사 앱에서 다시 확인해야 합니다.
- 같은 주문 ID의 멱등 처리 방식과 보존 기간은 토스증권·빗썸 API 정책에 따라 달라질 수 있습니다.

## 면책

이 프로젝트는 학습 및 개인 모니터링 목적으로 제작되었습니다.

- 투자 자문 또는 수익 보장 도구가 아닙니다.
- 관찰 종목은 매수 추천이 아닙니다.
- 기술지표와 백테스트는 미래 가격을 예측하거나 보장하지 않습니다.
- 주문 기능 사용과 투자 결과에 대한 책임은 사용자에게 있습니다.

## 라이선스

MIT License

## 만든 사람

[@juyoo0729](https://github.com/juyoo0729)
