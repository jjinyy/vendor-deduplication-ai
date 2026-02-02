"""
누락된 중복 가능성 분석
"""
import pandas as pd
from duplicate_detector_hybrid import DuplicateDetectorHybrid
from rapidfuzz import fuzz

# 결과 파일 로드
df = pd.read_excel('bio_vendor_merged_hybrid_20260114_150438.xlsx')

print("="*80)
print("누락된 중복 가능성 분석")
print("="*80)
print()

# 중복 그룹이 아닌 행들 중에서 유사한 이름을 가진 쌍 찾기
print("1. 중복 그룹이 아닌 행들 중 유사한 이름 분석")
print("-"*80)

# 중복 그룹이 아닌 행들
non_duplicate_df = df[df['_duplicate_count'].isna()].copy()

print(f"중복 그룹이 아닌 행 수: {len(non_duplicate_df):,}행")
print()

# 국가별로 그룹화하여 같은 국가 내에서 유사한 이름 찾기
detector = DuplicateDetectorHybrid(similarity_threshold=0.85, use_embedding=True)

# 샘플 분석 (처음 1000개 행만)
sample_size = min(1000, len(non_duplicate_df))
sample_df = non_duplicate_df.head(sample_size)

potential_misses = []
checked_pairs = set()

print(f"샘플 분석 중... (처음 {sample_size}개 행)")
print()

for i in range(len(sample_df)):
    if i % 100 == 0:
        print(f"  진행: {i}/{len(sample_df)}")
    
    row1 = sample_df.iloc[i]
    name1 = str(row1.get('공급업체명', ''))
    land1 = str(row1.get('Land', ''))
    
    if not name1 or name1 == 'nan':
        continue
    
    # 같은 국가 내에서만 비교
    same_country = sample_df[sample_df['Land'] == land1]
    
    for j in range(i+1, min(i+50, len(same_country))):  # 각 행마다 최대 50개씩만 비교
        row2 = same_country.iloc[j]
        name2 = str(row2.get('공급업체명', ''))
        
        if not name2 or name2 == 'nan':
            continue
        
        # 이미 확인한 쌍은 스킵
        pair_key = tuple(sorted([i, j]))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)
        
        # 빠른 이름 유사도 확인
        name_sim = fuzz.ratio(name1.upper(), name2.upper()) / 100.0
        
        # 이름 유사도가 높으면 (0.7 이상) 상세 분석
        if name_sim >= 0.70:
            # 실제 중복 판단
            is_dup, confidence = detector.are_duplicates(row1, row2)
            
            if not is_dup and name_sim >= 0.80:
                # 중복이 아니라고 판단되었지만 이름이 매우 유사한 경우
                potential_misses.append({
                    'name1': name1[:50],
                    'name2': name2[:50],
                    'name_sim': name_sim,
                    'land': land1,
                    'confidence': confidence
                })

print()
print(f"발견된 잠재적 누락: {len(potential_misses)}개")
print()

if potential_misses:
    print("상위 10개 잠재적 누락:")
    print("-"*80)
    for i, miss in enumerate(sorted(potential_misses, key=lambda x: x['name_sim'], reverse=True)[:10], 1):
        print(f"{i}. 이름 유사도: {miss['name_sim']:.3f}")
        print(f"   업체1: {miss['name1']}")
        print(f"   업체2: {miss['name2']}")
        print(f"   국가: {miss['land']}")
        print()

print("="*80)
print("결론:")
print(f"- 샘플 {sample_size}개 행 중 {len(potential_misses)}개의 잠재적 누락 발견")
print(f"- 전체 데이터에서 예상 누락: 약 {len(potential_misses) * len(non_duplicate_df) // sample_size}개")
print()
print("권장 사항:")
print("- 중복 판단 기준을 약간 완화하여 더 많은 중복을 탐지할 수 있습니다")
print("- 또는 현재 기준을 유지하고 수동 검토를 통해 누락된 중복을 확인할 수 있습니다")

