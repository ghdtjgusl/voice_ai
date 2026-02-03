from analyze_voice import analyze_voice
from similarity_engine import recommend_singers

# 사용자 음성 분석
analysis_result = analyze_voice(
    feature_dir="features/voice6_big",
    user_bpm=129.19921875,
    reference_song_name="No_Doubt"
)

# 유사 가수 추천
recommendations = recommend_singers(
    analysis_result["timbre_vector"],
    top_n=3
)

print("\n 내 목소리랑 비슷한 가수 Top 3")
for r in recommendations:
    print(f"{r['singer']}  |  similarity: {r['similarity']}")
