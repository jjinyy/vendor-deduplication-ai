"""
현재까지 발견된 중복 그룹을 기반으로 중간 결과 파일 생성
"""
import pandas as pd
import re
from data_loader import DataLoader
from merger import DataMerger
from datetime import datetime

# 로그 파일 읽기
log_file = 'process_log_hybrid.txt'
print(f"로그 파일 읽기: {log_file}")

# 중복 발견 메시지 파싱
duplicate_pairs = []
with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        if '[중복 발견]' in line:
            # 예: [중복 발견] ADITIYA BIRLA CHEMICALS (THAILAND) <-> ADITYA BIRLA CHEMICALS (THAILAND) L (신뢰도: 0.97)
            match = re.search(r'\[중복 발견\]\s+(.+?)\s+<->\s+(.+?)\s+\(신뢰도:', line)
            if match:
                name1 = match.group(1).strip()
                name2 = match.group(2).strip()
                duplicate_pairs.append((name1, name2))
                print(f"발견된 중복: {name1} <-> {name2}")

print(f"\n총 발견된 중복 쌍: {len(duplicate_pairs)}개")

if len(duplicate_pairs) == 0:
    print("아직 발견된 중복이 없습니다.")
    exit(0)

# 원본 데이터 로드
print("\n원본 데이터 로드 중...")
loader = DataLoader('bio_vendor.csv')
df = loader.load()
print(f"로드 완료: {len(df):,}행")

# 중복 그룹 생성 (이름으로 매칭)
duplicate_groups = []
used_indices = set()

for name1, name2 in duplicate_pairs:
    # 이름으로 인덱스 찾기
    indices1 = df[df['공급업체명'].astype(str).str.contains(name1[:30], case=False, na=False, regex=False)].index.tolist()
    indices2 = df[df['공급업체명'].astype(str).str.contains(name2[:30], case=False, na=False, regex=False)].index.tolist()
    
    # 가장 유사한 인덱스 찾기 (정확한 매칭 우선)
    matched_indices = []
    for idx1 in indices1:
        if idx1 not in used_indices:
            name1_actual = str(df.loc[idx1, '공급업체명'])
            if name1[:30].upper() in name1_actual.upper() or name1_actual.upper() in name1[:30].upper():
                matched_indices.append(idx1)
                break
    
    for idx2 in indices2:
        if idx2 not in used_indices:
            name2_actual = str(df.loc[idx2, '공급업체명'])
            if name2[:30].upper() in name2_actual.upper() or name2_actual.upper() in name2[:30].upper():
                matched_indices.append(idx2)
                break
    
    if len(matched_indices) >= 2:
        duplicate_groups.append(matched_indices)
        used_indices.update(matched_indices)
        print(f"그룹 생성: {len(matched_indices)}개 행")

print(f"\n생성된 중복 그룹: {len(duplicate_groups)}개")

if len(duplicate_groups) == 0:
    print("중복 그룹을 생성할 수 없습니다. 로그에서 정확한 인덱스를 찾을 수 없습니다.")
    exit(0)

# 그룹번호 부여
merger = DataMerger()
df_result = merger.merge_duplicates(df, duplicate_groups)

# _merged_from_indices 컬럼 제거
if '_merged_from_indices' in df_result.columns:
    df_result = df_result.drop(columns=['_merged_from_indices'])

# 중간 결과 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'중간결과_{len(duplicate_groups)}개그룹_{timestamp}.xlsx'
print(f"\n중간 결과 저장 중: {output_file}")

df_result.to_excel(output_file, index=False)
print(f"저장 완료: {output_file}")
print(f"총 {len(df_result):,}행, 중복 그룹 {len(duplicate_groups)}개")


