"""
audio/ 폴더의 모든 wav 파일에 대해
1) Demucs로 MR 제거
2) 원본 음원 → tempo 분석
3) MR 제거된 보컬 → pitch / volume 분석
4) 전처리 포함 (22050Hz, mono, trim, noise reduction)
5) 결과를 results/summary_features.json 에 저장
"""

import os
import json
import subprocess
import librosa
import numpy as np

# =====================
# 경로 설정
# =====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_PATH = os.path.join(RESULT_DIR, "summary_features.json")

TARGET_SR = 22050  # 샘플레이트 통일


# =====================
# Demucs MR 제거
# =====================

def separate_vocals(audio_path):
    """
    Demucs 실행 → vocals.mp3 경로 반환
    """
    subprocess.run(
        [
            "python", "-m", "demucs",
            "--two-stems=vocals",
            "--mp3",               # torchcodec 충돌 회피
            "-o", TEMP_DIR,
            audio_path
        ],
        check=True
    )

    base = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(
        TEMP_DIR, "htdemucs", base, "vocals.mp3"
    )

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"vocals.mp3 생성 실패: {vocals_path}")

    return vocals_path


# =====================
# 전처리 함수
# =====================

def preprocess_audio(y, sr):
    """
    1) Trim (무음 제거)
    2) 간단한 Noise Reduction (Spectral Gating)
    """

    # --- 무음 제거 ---
    y, _ = librosa.effects.trim(y, top_db=25)

    # --- Noise Reduction (저에너지 주파수 제거) ---
    stft = librosa.stft(y)
    magnitude, phase = np.abs(stft), np.angle(stft)

    noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
    magnitude_denoised = np.maximum(magnitude - noise_profile, 0)

    y_denoised = librosa.istft(
        magnitude_denoised * np.exp(1j * phase)
    )

    return y_denoised


# =====================
# Feature 추출
# =====================

def extract_features(original_audio, vocals_audio):
    """
    tempo → 원본 음원
    pitch / volume → MR 제거된 보컬
    """

    # =================
    # Tempo (원본 음원)
    # =================

    y_org, sr_org = librosa.load(
        original_audio,
        sr=TARGET_SR,
        mono=True
    )

    y_org, _ = librosa.effects.trim(y_org)
    tempo, _ = librosa.beat.beat_track(y=y_org, sr=sr_org)

    # =================
    # Pitch / Volume (보컬)
    # =================

    y_v, sr_v = librosa.load(
        vocals_audio,
        sr=TARGET_SR,
        mono=True
    )

    y_v = preprocess_audio(y_v, sr_v)

    # --- Pitch ---
    f0, _, _ = librosa.pyin(
        y_v,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    f0_clean = f0[~np.isnan(f0)]
    pitch_avg = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0

    # --- Volume ---
    rms = librosa.feature.rms(y=y_v)
    volume_rms = float(np.mean(rms))

    return {
        "tempo_bpm": float(tempo),
        "pitch_hz_avg": pitch_avg,
        "volume_rms_avg": volume_rms
    }


# =====================
# 메인 실행
# =====================

def main():
    results = []

    wav_files = [
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    ]

    for fname in wav_files:
        audio_path = os.path.join(AUDIO_DIR, fname)
        print(f"\n 처리 중: {fname}")

        try:
            vocals_path = separate_vocals(audio_path)
            features = extract_features(audio_path, vocals_path)

            result = {
                "file_name": fname,
                **features
            }

            results.append(result)
            print(json.dumps(result, indent=4, ensure_ascii=False))

        except Exception as e:
            print(f" 실패: {fname}")
            print(e)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n 모든 MR 제거 음성 분석 완료")
    print(f" 결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
