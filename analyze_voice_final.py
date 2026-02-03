import os
import json

from extract_basic_features import extract_single_wav
from analyze_voice import analyze_voice
from similarity_engine import recommend_singers


# ===============================
# 1. 점수 기반 자동 피드백
# ===============================
def generate_feedback(scores):
    feedback = []

    # Pitch
    if scores["pitch"] >= 90:
        feedback.append("음정이 원곡과 거의 일치합니다.")
    elif scores["pitch"] >= 70:
        feedback.append("전반적으로 안정적인 음정입니다.")
    else:
        feedback.append("음정 기복이 있어 연습이 필요합니다.")

    # Tempo
    if scores["tempo"] >= 90:
        feedback.append("박자를 정확하게 잘 지켰습니다.")
    else:
        feedback.append("박자가 다소 흔들립니다.")

    # Volume
    if scores["volume"] >= 80:
        feedback.append("발성이 시원하고 힘이 있습니다.")
    else:
        feedback.append("조금 더 자신 있게 소리를 내도 좋겠습니다.")

    return " ".join(feedback)


# ===============================
# 2. 서버 호출용 최종 함수
# ===============================
def analyzeVoice(
    wav_path,
    reference_song,
    user_bpm,
    top_n=3
):
    # -------------------------------
    # 0) 파일 존재 확인
    # -------------------------------
    if not os.path.exists(wav_path):
        return {"error": "음성 파일이 존재하지 않습니다."}

    # -------------------------------
    # 1) wav 1개만 feature 추출
    # -------------------------------
    extract_single_wav(wav_path)

    voice_name = os.path.splitext(os.path.basename(wav_path))[0]
    feature_dir = os.path.join("features", voice_name)

    if not os.path.exists(feature_dir):
        return {"error": "feature 생성 실패"}

    # -------------------------------
    # 2) 분석
    # -------------------------------
    analysis_result = analyze_voice(
        feature_dir=feature_dir,
        user_bpm=user_bpm,
        reference_song_name=reference_song
    )

    # -------------------------------
    # 3) 무음 예외 처리 (요청사항)
    # -------------------------------
    rms = analysis_result["analysis_values"]["volume_rms_avg"]
    if rms < 0.01:
        return {"error": "음성이 거의 감지되지 않습니다."}

    # -------------------------------
    # 4) 추천
    # -------------------------------
    recommendations = recommend_singers(
        analysis_result["timbre_vector"],
        top_n=top_n
    )

    # -------------------------------
    # 5) 피드백
    # -------------------------------
    feedback_text = generate_feedback(
        analysis_result["scores"]
    )

    # -------------------------------
    # 6) 최종 JSON
    # -------------------------------
    return {
        "scores": analysis_result["scores"],
        "analysis_values": analysis_result["analysis_values"],
        "feedback": feedback_text,
        "recommendations": recommendations
    }


# ===============================
# 3. 단독 테스트
# ===============================
if __name__ == "__main__":
    result = analyzeVoice(
        wav_path="audio/voice6_big.wav",
        reference_song="No_Doubt",
        user_bpm=129.2,
        top_n=3
    )

    print("\n===== 분석 결과 =====")
    print(json.dumps(result, indent=4, ensure_ascii=False))
