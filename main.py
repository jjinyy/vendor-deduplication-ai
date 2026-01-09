"""
협력사 중복 탐지 및 병합 메인 애플리케이션
"""
import pandas as pd
import argparse
import os
from datetime import datetime
from data_loader import DataLoader
from duplicate_detector import DuplicateDetector
from merger import DataMerger


def print_summary(df_original: pd.DataFrame, df_merged: pd.DataFrame, 
                  duplicate_groups: list):
    """결과 요약 출력"""
    print("\n" + "="*60)
    print("중복 탐지 및 병합 결과 요약")
    print("="*60)
    print(f"원본 데이터 행 수: {len(df_original)}")
    print(f"병합 후 행 수: {len(df_merged)}")
    print(f"제거된 중복 행 수: {len(df_original) - len(df_merged)}")
    print(f"중복 그룹 수: {len(duplicate_groups)}")
    
    if duplicate_groups:
        print("\n중복 그룹 상세:")
        for i, group in enumerate(duplicate_groups, 1):
            print(f"\n그룹 {i} ({len(group)}개 행):")
            for idx in group:
                name = df_original.loc[idx, '공급업체명'] if '공급업체명' in df_original.columns else 'N/A'
                code = df_original.loc[idx, '공급업체코드'] if '공급업체코드' in df_original.columns else 'N/A'
                land = df_original.loc[idx, 'Land'] if 'Land' in df_original.columns else 'N/A'
                print(f"  - 인덱스 {idx}: [{code}] {name} ({land})")


def main():
    parser = argparse.ArgumentParser(
        description='협력사 데이터의 중복을 탐지하고 병합합니다.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='입력 파일 경로 (CSV 또는 Excel)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='출력 파일 경로 (지정하지 않으면 자동 생성)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.85,
        help='유사도 임계값 (0.0 ~ 1.0, 기본값: 0.85)'
    )
    parser.add_argument(
        '-s', '--strategy',
        type=str,
        choices=['best', 'first', 'latest'],
        default='best',
        help='병합 전략: best(가장 완전한 데이터), first(첫 번째), latest(최신)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='요약 정보만 출력하고 파일 저장하지 않음'
    )
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input_file):
        print(f"오류: 파일을 찾을 수 없습니다: {args.input_file}")
        return
    
    print(f"파일 로딩 중: {args.input_file}")
    
    # 데이터 로드
    loader = DataLoader(args.input_file)
    df_original = loader.load()
    
    print(f"로드된 데이터: {len(df_original)}행, {len(df_original.columns)}열")
    
    # 중복 탐지
    print("\n중복 탐지 중...")
    print(f"데이터 크기: {len(df_original)}행")
    detector = DuplicateDetector(similarity_threshold=args.threshold)
    duplicate_groups = detector.detect_duplicates(df_original, show_progress=True)
    
    print(f"탐지된 중복 그룹: {len(duplicate_groups)}개")
    
    if not duplicate_groups:
        print("\n중복이 발견되지 않았습니다.")
        return
    
    # 데이터 병합
    print("\n데이터 병합 중...")
    merger = DataMerger(merge_strategy=args.strategy)
    df_merged = merger.merge_duplicates(df_original, duplicate_groups)
    
    # 결과 요약 출력
    print_summary(df_original, df_merged, duplicate_groups)
    
    # 결과 저장
    if not args.summary_only:
        if args.output is None:
            # 자동으로 출력 파일명 생성
            base_name = os.path.splitext(args.input_file)[0]
            ext = os.path.splitext(args.input_file)[1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"{base_name}_merged_{timestamp}{ext}"
        
        print(f"\n결과 저장 중: {args.output}")
        
        file_ext = os.path.splitext(args.output)[1].lower()
        if file_ext == '.csv':
            df_merged.to_csv(args.output, index=False, encoding='utf-8-sig')
        elif file_ext in ['.xlsx', '.xls']:
            df_merged.to_excel(args.output, index=False)
        else:
            df_merged.to_csv(args.output, index=False, encoding='utf-8-sig')
        
        print(f"저장 완료: {args.output}")
    
    print("\n처리 완료!")


if __name__ == '__main__':
    main()

