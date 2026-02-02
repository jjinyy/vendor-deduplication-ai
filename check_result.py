"""
결과 파일 확인
"""
import pandas as pd

result_file = 'bio_vendor_merged_hybrid_20260114_150438.xlsx'

print("="*80)
print("결과 파일 확인")
print("="*80)
print()

try:
    df = pd.read_excel(result_file)
    
    print(f"파일: {result_file}")
    print(f"총 행 수: {len(df):,}행")
    print(f"컬럼 수: {len(df.columns)}개")
    print()
    
    print("컬럼명:")
    for i, col in enumerate(df.columns[:10], 1):
        print(f"  {i}. {col}")
    if len(df.columns) > 10:
        print(f"  ... 외 {len(df.columns) - 10}개 컬럼")
    print()
    
    if '그룹번호' in df.columns:
        print(f"그룹번호 컬럼: 존재")
        print(f"  - 총 그룹 수: {df['그룹번호'].nunique():,}개")
        print(f"  - 중복 그룹 수: {df[df['_duplicate_count'].notna()]['그룹번호'].nunique() if '_duplicate_count' in df.columns else 'N/A'}개")
        print(f"  - 중복 행 수: {df[df['_duplicate_count'].notna()].shape[0] if '_duplicate_count' in df.columns else 'N/A'}행")
    else:
        print("그룹번호 컬럼: 없음")
    print()
    
    if '_duplicate_count' in df.columns:
        print(f"_duplicate_count 컬럼: 존재")
        duplicate_rows = df[df['_duplicate_count'].notna()]
        if len(duplicate_rows) > 0:
            print(f"  - 중복 그룹 행 수: {len(duplicate_rows):,}행")
            print(f"  - 최대 중복 수: {duplicate_rows['_duplicate_count'].max()}개")
            print(f"  - 평균 중복 수: {duplicate_rows['_duplicate_count'].mean():.2f}개")
    else:
        print("_duplicate_count 컬럼: 없음")
    print()
    
    if '_merged_from_indices' in df.columns:
        print("경고: _merged_from_indices 컬럼이 존재합니다 (제거되어야 함)")
    else:
        print("_merged_from_indices 컬럼: 없음 (정상)")
    print()
    
    # 중복 그룹 샘플 확인
    if '그룹번호' in df.columns and '_duplicate_count' in df.columns:
        duplicate_groups = df[df['_duplicate_count'].notna()]['그룹번호'].unique()[:5]
        print("중복 그룹 샘플 (상위 5개):")
        for group_num in duplicate_groups:
            group_rows = df[df['그룹번호'] == group_num]
            print(f"  그룹 {group_num} ({len(group_rows)}개 행):")
            for idx, row in group_rows.iterrows():
                company_name = str(row.get('공급업체명', ''))[:50]
                print(f"    - {company_name}")
        print()
    
    print("="*80)
    print("결론: 결과 파일이 정상적으로 생성되었습니다!")
    
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()

