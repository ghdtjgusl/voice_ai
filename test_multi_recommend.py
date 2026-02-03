import numpy as np
from analyze_voice import build_timbre_vector
from similarity_engine import recommend_singers

# ===============================
# 1. 내 목소리 feature 폴더들 (실제 이름)
# ===============================
my_voice_dirs = [
    "features/voice1_mrX",
    "features/voice2_mr",
    "features/voice3_slow",
    "features/voice4_fast",
    "features/voice5_small",
    "features/voice6_big"
]

# ===============================
# 2. 각 목소리의 timbre_vector 추출
# ===============================
vectors = []

for d in my_voice_dirs:
    vec = build_timbre_vector(d)
    vectors.append(vec)
    print(f"✓ {d} 처리 완료")

print(f"\n사용된 목소리 개수: {len(vectors)}")

# ===============================
# 3. 평균 음색 벡터 생성 (핵심)
# ===============================
mean_vector = np.mean(vectors, axis=0).tolist()

# ===============================
# 4. 가수 Top-3 추천
# ===============================
recommendations = recommend_singers(
    mean_vector,
    top_n=3
)

print("\n 내 목소리 여러 개 기준 Top 3 가수")
for r in recommendations:
    print(f"{r['singer']} | similarity: {r['similarity']}")
