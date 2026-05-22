from sentence_transformers import SentenceTransformer, util
from naver_finance_feed import fetch_naver_finance_news

# 1) 뉴스 가져오기
news_list = fetch_naver_finance_news(max_items=30)
if not news_list:
    print("뉴스를 가져오지 못했습니다. 네트워크 상태 또는 페이지 구조를 확인하세요.")
    raise SystemExit(1)

titles = [news["title"] for news in news_list]
texts_for_embedding = [news["title"] + " " + news["summary"] for news in news_list]
print(f"가져온 뉴스: {len(titles)}건\n")

# 2) 모델 로드  ← 이게 임베딩보다 위에 있어야 함!
print("모델 로딩 중...")
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 3) 임베딩  ← model이 위에서 만들어진 다음에 와야 함
embeddings = model.encode(texts_for_embedding, convert_to_tensor=True, normalize_embeddings=True)

# 4) 클러스터링
clusters = util.community_detection(
    embeddings, min_community_size=2, threshold=0.55
)

# 5) 결과 출력
print("\n" + "=" * 50)
print(f"  오늘 묶인 주제: {len(clusters)}개")
print("=" * 50)

for rank, cluster in enumerate(clusters, start=1):
    print(f"\n[주제 {rank}]  기사 {len(cluster)}건이 한 덩어리로 묶임")
    for idx in cluster:
        print(f"   - {titles[idx]}")

grouped = {idx for c in clusters for idx in c}
loners = [titles[i] for i in range(len(titles)) if i not in grouped]
if loners:
    print(f"\n[묶이지 않은 단독 기사] {len(loners)}건")
    for t in loners:
        print(f"   - {t}")