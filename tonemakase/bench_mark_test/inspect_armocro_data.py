"""
Deep-Armocromia 데이터셋 검증 스크립트
실행: python inspect_armocromia.py
"""
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path("./Deep-Armocromia/dataset")
CSV_PATH = BASE_DIR / "annotations.csv"

# ====================================================
# 1. CSV 기본 정보
# ====================================================
print("\n" + "="*60)
print("1. CSV 기본 정보")
print("="*60)
df = pd.read_csv(CSV_PATH)
print(f"  전체 행 수     : {len(df)}")
print(f"  컬럼 목록      : {list(df.columns)}")
print(f"\n  첫 3행 샘플:")
print(df.head(3).to_string())

# ====================================================
# 2. partition 분포
# ====================================================
print("\n" + "="*60)
print("2. partition 분포")
print("="*60)
print(df["partition"].value_counts().to_string())

# ====================================================
# 3. class 필드 유니크 값
# ====================================================
print("\n" + "="*60)
print("3. class 필드 유니크 값 (이탈리아어 → 영어 매핑 확인)")
print("="*60)
SEASON_MAP = {"primavera": "spring", "estate": "summer",
              "autunno": "autumn", "inverno": "winter"}
class_counts = df["class"].value_counts()
print(f"  유니크 class 값: {list(class_counts.index)}")
print()
for cls, cnt in class_counts.items():
    mapped = SEASON_MAP.get(str(cls).strip().lower(), "❌ 매핑 없음!")
    print(f"  '{cls}' : {cnt}개  →  {mapped}")

# ====================================================
# 4. SEASON_MAP 누락 class 탐지
# ====================================================
print("\n" + "="*60)
print("4. SEASON_MAP에 없는 class 값 (매핑 오류 탐지)")
print("="*60)
unknown = df[~df["class"].str.strip().str.lower().isin(SEASON_MAP.keys())]
print(f"  매핑 불가 샘플 수: {len(unknown)}개")
if len(unknown) > 0:
    print(unknown[["partition", "class", "path_rgb_original"]].head(10).to_string())

# ====================================================
# 5. 이미지 파일 실제 존재 여부 확인 (test 파티션)
# ====================================================
print("\n" + "="*60)
print("5. test 파티션 이미지 파일 존재 여부")
print("="*60)
test_df = df[df["partition"] == "test"]
missing = []
found = 0
for _, row in test_df.iterrows():
    img_path = BASE_DIR / str(row["path_rgb_original"])
    if img_path.exists():
        found += 1
    else:
        missing.append(str(row["path_rgb_original"]))

print(f"  test 샘플 수   : {len(test_df)}")
print(f"  이미지 존재    : {found}개")
print(f"  이미지 없음    : {len(missing)}개")
if missing:
    print(f"  누락 예시      : {missing[:5]}")

# ====================================================
# 6. test 파티션 클래스별 분포 (불균형 확인)
# ====================================================
print("\n" + "="*60)
print("6. test 파티션 클래스별 샘플 수 (불균형 확인)")
print("="*60)
test_class = test_df["class"].value_counts()
total_test = len(test_df)
for cls, cnt in test_class.items():
    mapped = SEASON_MAP.get(str(cls).strip().lower(), "?")
    pct = cnt / total_test * 100
    print(f"  {cls:<12} ({mapped:<6}) : {cnt:>4}개  ({pct:.1f}%)")

# ====================================================
# 7. 중복 이미지 경로 확인
# ====================================================
print("\n" + "="*60)
print("7. 중복 이미지 경로 (같은 이미지가 여러 행에 있는지)")
print("="*60)
dup = df[df.duplicated(subset=["path_rgb_original"], keep=False)]
print(f"  중복 경로 샘플 수: {len(dup)}개")
if len(dup) > 0:
    print(dup[["partition","class","path_rgb_original"]].head(6).to_string())

print("\n완료!")