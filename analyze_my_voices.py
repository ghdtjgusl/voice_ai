import os
import json

from analyze_voice import analyze_voice
from similarity_engine import recommend_singers


# ===============================
# 성량 점수 (절대 RMS 기준)
# ===============================
def score_volume(rms):
    if rms >= 0.08:
        return 90
    elif rms >= 0.05:
        return 75
    elif rms >= 0.03:
        return 60
    else:
        return 40


# ===============================
# 성량 피드백 (절대 RMS 기준)
# ===============================
def volume_feedback(rms):
    if rms >= 0.08:
        return "성량이 충분하고 힘 있는 발성입니다."
    elif rms >= 0.05:
        return "안정적인 성량으로 잘 불렀습니다."
    else:
        return "조금 더 자신 있게 소리를 내도 좋겠습니다."


# ===============================
# 최종 피드백 생성
# ===============================
def generate_feedback(scores, analysis_values):
    feedback = []

    # 음정
    if scores["pitch"] >= 90:
        feedback.append("음정이 매우 안정적입니다.")
    elif scores["pitch"] >= 70:
        feedback.append("전반적으로 음정은 무난합니다.")
    else:
        feedback.append("음정 기복이 있어 연습이 필요합니다.")

    # 박자
    if scores["tempo"] >= 90:
        feedback.append("박자를 정확하게 잘 지켰습니다.")
    else:
        feedback.append("박자가 다소 흔들립니다.")

    # 성량 (RMS 직접 사용)
    feedback.append(
        volume_feedback(analysis_values["volume_rms_avg"])
    )

    return " ".join(feedback)


# ===============================
# feature 폴더 하나 분석
# ===============================
def analyze_one_voice(feature_dir, reference_song, user_bpm, top_n=3):

    result = analyze_voice(
        feature_dir=feature_dir,
        user_bpm=user_bpm,
        reference_song_name=reference_song
    )

    # volume 점수 교체 (핵심)
    rms = result["analysis_values"]["volume_rms_avg"]
    result["scores"]["volume"] = score_volume(rms)

    feedback = generate_feedback(
        result["scores"],
        result["analysis_values"]
    )

    recommendations = recommend_singers(
        result["timbre_vector"],
        top_n=top_n
    )

    return {
        "voice_name": os.path.basename(feature_dir),
        "scores": result["scores"],
        "analysis_values": result["analysis_values"],
        "feedback": feedback,
        "recommendations": recommendations
    }


# ===============================
# 내 목소리 여러 개 평가
# ===============================
if __name__ == "__main__":

    my_voice_dirs = [
        "features/voice1_mrX",
        "features/voice2_mr",
        "features/voice3_slow",
        "features/voice4_fast",
        "features/voice5_small",
        "features/voice6_big",
    ]

    all_results = []

    for vdir in my_voice_dirs:
        print(f"\n===== {os.path.basename(vdir)} 분석 중 =====")

        result = analyze_one_voice(
            feature_dir=vdir,
            reference_song="No_Doubt",
            user_bpm=129.2,
            top_n=3
        )

        all_results.append(result)
        print(json.dumps(result, indent=4, ensure_ascii=False))

    # 전체 결과 저장
    with open("my_voice_feedback.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\n 모든 내 목소리 평가 완료")
