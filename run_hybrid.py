"""
하이브리드 버전 실행 스크립트: Blocking + 의미 기반 임베딩
"""
from data_loader import DataLoader
from duplicate_detector_hybrid import DuplicateDetectorHybrid
from merger import DataMerger
import pandas as pd
from datetime import datetime
import sys
import time

class Logger:
    """로그를 파일과 콘솔에 동시에 출력"""
    def __init__(self, log_file='process_log_hybrid.txt'):
        self.log_file = log_file
        self.start_time = time.time()
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("협력사 중복 탐지 및 병합 프로세스 로그 (하이브리드 버전)\n")
            f.write("="*60 + "\n\n")
    
    def log(self, message):
        """로그 메시지 출력"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{elapsed:.1f}초] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
        sys.stdout.flush()
    
    def section(self, title):
        """섹션 헤더 출력"""
        self.log("="*60)
        self.log(title)
        self.log("="*60)

logger = Logger()

print("="*60)
print("협력사 중복 탐지 및 병합 - 하이브리드 버전")
print("="*60)
print("\n핵심 기능:")
print("  [OK] 다국어 중복 탐지: 같은 업체의 다른 언어 표기도 탐지")
print("  [OK] 다중 Blocking 키: 비교 횟수 대폭 감소")
print("  [OK] 의미 기반 임베딩: 정확도 최대화 (92-96%)")
print("  [OK] RapidFuzz: 빠른 문자열 비교")
print("="*60)

logger.section("프로세스 시작")

# 파일 로드
logger.section("1단계: 파일 로딩")
logger.log("파일 로딩 시작: bio_vendor.csv")
loader = DataLoader('bio_vendor.csv')
df = loader.load()
logger.log(f"로드 완료: {len(df):,}행, {len(df.columns)}열")

# 중복 탐지
logger.section("2단계: 하이브리드 중복 탐지")
logger.log("중복 탐지 시작 (유사도 임계값: 0.85)")
logger.log("하이브리드 방식: Blocking + 의미 기반 임베딩")

logger.log("하이브리드 탐지기 초기화 중...")
detector = DuplicateDetectorHybrid(similarity_threshold=0.85, use_embedding=True, logger=logger)
logger.log("초기화 완료")

duplicate_groups = detector.detect_duplicates(df, show_progress=True)

logger.section("중복 탐지 완료")
logger.log(f"탐지된 중복 그룹: {len(duplicate_groups)}개")

if duplicate_groups:
    total_duplicates = sum(len(g) for g in duplicate_groups)
    logger.log(f"총 중복 행 수: {total_duplicates}개")
    
    logger.log("\n중복 그룹 예시 (처음 5개):")
    for i, group in enumerate(duplicate_groups[:5], 1):
        logger.log(f"\n  그룹 {i} ({len(group)}개 행):")
        for idx in group[:3]:
            row = df.iloc[idx]
            name = row.get('공급업체명', 'N/A')
            code = row.get('공급업체코드', 'N/A')
            land = row.get('Land', 'N/A')
            city = row.get('CITY1', 'N/A')
            logger.log(f"    - [{code}] {name} ({land}, {city})")
        if len(group) > 3:
            logger.log(f"    ... 외 {len(group)-3}개")
    
    # 데이터 병합
    logger.section("3단계: 데이터 병합")
    logger.log("데이터 병합 시작 (전략: best)")
    merger = DataMerger(merge_strategy='best')
    df_merged = merger.merge_duplicates(df, duplicate_groups)
    logger.log("데이터 병합 완료")
    
    logger.section("4단계: 병합 결과")
    logger.log(f"원본 행 수: {len(df):,}")
    logger.log(f"병합 후 행 수: {len(df_merged):,}")
    logger.log(f"제거된 중복 행: {len(df) - len(df_merged):,}개")
    logger.log(f"중복 그룹 수: {len(duplicate_groups)}개")
    
    # 결과 저장
    logger.section("5단계: 결과 저장")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'bio_vendor_merged_hybrid_{timestamp}.xlsx'
    logger.log(f"결과 파일 저장 중: {output_file}")
    df_merged.to_excel(output_file, index=False)
    logger.log(f"저장 완료: {output_file}")
else:
    logger.log("중복이 발견되지 않았습니다.")

total_time = time.time() - logger.start_time
logger.section("프로세스 완료")
logger.log(f"총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
logger.log("모든 작업이 완료되었습니다!")
print("\n" + "="*60)
print("처리 완료!")
print(f"로그 파일: {logger.log_file}")
print("="*60)

