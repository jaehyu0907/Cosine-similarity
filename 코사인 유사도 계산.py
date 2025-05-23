from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 한국어용 사전학습 모델 불러오기
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 사용자 입력
sentence1 = input("첫 번째 문장을 입력하세요: ")
sentence2 = input("두 번째 문장을 입력하세요: ")

# 문장을 벡터로 변환
vec1 = model.encode(sentence1)
vec2 = model.encode(sentence2)

# 코사인 유사도 계산
similarity = cosine_similarity([vec1], [vec2])[0][0]

# 결과 출력
print(f"두 문장의 코사인 유사도: {similarity:.3f}")
