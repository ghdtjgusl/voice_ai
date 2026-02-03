import os
import json
from analyze_voice import build_timbre_vector

# ===============================
# 경로 설정
# ===============================
FEATURE_BASE = "features"
SINGER_DB_DIR = "singer_db"

os.makedirs(SINGER_DB_DIR, exist_ok=True)

# ===============================
# 가수 목록
# key   : singer_db에 저장될 이름
# value : features 폴더 이름
# ===============================
singers = {
    "ChoiYuri": "choiyuri",
    "Hwasa": "hwasa",
    "IU": "iu",
    "KimNaYoung": "kimnayoung",
    "Taeyeon": "taeyeon"
}

# ===============================
# singer_db 생성
# ===============================
for singer_name, feature_folder in singers.items():
    feature_dir = os.path.join(FEATURE_BASE, feature_folder)

    if not os.path.exists(feature_dir):
        print(f" feature 폴더 없음: {feature_dir}")
        continue

    timbre_vector = build_timbre_vector(feature_dir)

    data = {
        "singer": singer_name,
        "timbre_vector": timbre_vector
    }

    out_path = os.path.join(SINGER_DB_DIR, f"{singer_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f" {singer_name} singer_db 생성 완료")

print("\n 모든 가수 DB 생성 완료")
