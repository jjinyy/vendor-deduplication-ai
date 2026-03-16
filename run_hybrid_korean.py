"""
한국 업체 전용 - 하이브리드 중복 탐지 실행 스크립트
- 입력: 공급업체코, 공급업체명, 주소 (한국 형식 CSV/Excel)
- 기존 run_hybrid.py / duplicate_detector_hybrid 는 해외 업체용으로 유지
"""
from src.data_loader_korean import DataLoaderKorean
from src.duplicate_detector_hybrid import DuplicateDetectorHybrid
from src.merger import DataMerger
import pandas as pd
from datetime import datetime
import sys
import time
import os

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 기본 입력 파일: data/ 우선, 없으면 루트 (기준정보는 data/에 두는 것을 권장)
_DEFAULT_NAME = 'FOOD 중복업체.xlsx'
DEFAULT_INPUT_FILE = os.path.join('data', _DEFAULT_NAME) if os.path.exists(os.path.join('data', _DEFAULT_NAME)) else _DEFAULT_NAME


class Logger:
    """로그를 파일과 콘솔에 동시에 출력"""
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = os.path.join(OUTPUT_DIR, 'process_log_hybrid_korean.txt')
        self.log_file = log_file
        self.start_time = time.time()
        with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
            f.write("="*60 + "\n")
            f.write("협력사 중복 탐지 및 병합 프로세스 로그 (한국 업체 전용)\n")
            f.write("="*60 + "\n\n")

    def log(self, message):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            safe_message = str(message)
        except Exception:
            try:
                safe_message = repr(message)
            except Exception:
                safe_message = "[인코딩 오류: 메시지를 표시할 수 없음]"
        try:
            safe_message = safe_message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except Exception:
            safe_message = "[인코딩 오류]"
        log_message = f"[{timestamp}] [{elapsed:.1f}초] {safe_message}"
        try:
            print(log_message)
        except UnicodeEncodeError:
            try:
                print(log_message.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            except Exception:
                print(f"[{timestamp}] [{elapsed:.1f}초] [로그 출력 오류]")
        try:
            with open(self.log_file, 'a', encoding='utf-8', errors='replace') as f:
                f.write(log_message + "\n")
                f.flush()
        except Exception:
            pass
        sys.stdout.flush()

    def section(self, title):
        self.log("="*60)
        self.log(title)
        self.log("="*60)


logger = Logger()

print("="*60)
print("협력사 중복 탐지 및 병합 - 한국 업체 전용")
print("="*60)
print("\n입력 형식: 공급업체코, 공급업체명, 주소")
print("  [OK] 한국 주소 파싱 (시/군/구 추출)")
print("  [OK] 동일 하이브리드 엔진 (Blocking + 임베딩)")
print("  [OK] STEP1/STEP2 그룹번호 부여")
print("="*60)

logger.section("프로세스 시작")

# 입력 파일 (스크립트 인자 또는 기본값)
input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE

logger.section("1단계: 파일 로딩")
logger.log(f"파일 로딩 시작: {input_file}")
loader = DataLoaderKorean(input_file)
df = loader.load()
logger.log(f"로드 완료: {len(df):,}행, {len(df.columns)}열 (한국 스키마 매핑됨)")

logger.section("2단계: 하이브리드 중복 탐지")
logger.log("중복 탐지 시작 (유사도 임계값: 0.85)")
logger.log("하이브리드 방식: Blocking + 의미 기반 임베딩 (한국 업체용)")

logger.log("하이브리드 탐지기 초기화 중...")
try:
    detector = DuplicateDetectorHybrid(similarity_threshold=0.85, use_embedding=True, logger=logger)
    logger.log("초기화 완료")

    checkpoint_file = os.path.join(OUTPUT_DIR, 'checkpoint_ann_korean.pkl')
    if os.path.exists(checkpoint_file):
        logger.log(f"[체크포인트 발견] {checkpoint_file} 이전 진행부터 재개합니다.")

    candidate_groups, final_groups, match_info = detector.detect_duplicates(
        df,
        show_progress=True,
        save_intermediate_at=10,
        candidate_mode='ann',
        checkpoint_file=checkpoint_file,
        checkpoint_interval=1000,
        output_dir=OUTPUT_DIR,
    )
except Exception as e:
    logger.log(f"[오류] 중복 탐지 중 오류 발생: {e}")
    import traceback
    logger.log(traceback.format_exc())
    candidate_groups = []
    final_groups = []
    match_info = {}

logger.section("중복 탐지 완료")
logger.log(f"STEP1 후보 그룹: {len(candidate_groups)}개")
logger.log(f"STEP2 최종 중복 그룹: {len(final_groups)}개")

if final_groups:
    total_duplicates = sum(len(g) for g in final_groups)
    logger.log(f"총 중복 행 수: {total_duplicates}개")

    logger.log("\n최종 중복 그룹 예시 (처음 5개):")
    for i, group in enumerate(final_groups[:5], 1):
        logger.log(f"\n  그룹 {i} ({len(group)}개 행):")
        for idx in group[:3]:
            row = df.loc[idx] if idx in df.index else df.iloc[idx]
            name = row.get('공급업체명', 'N/A')
            code = row.get('공급업체코드', 'N/A')
            land = row.get('Land', 'N/A')
            city = row.get('CITY1', 'N/A')
            logger.log(f"    - [{code}] {name} ({land}, {city})")
        if len(group) > 3:
            logger.log(f"    ... 외 {len(group)-3}개")

    logger.section("3단계: 데이터 병합")
    logger.log("데이터 병합 시작 (2단계 구조)")
    merger = DataMerger(merge_strategy='best')
    df_merged = merger.merge_duplicates_2step(df, candidate_groups, final_groups, match_info)
    logger.log("데이터 병합 완료")

    logger.section("4단계: 그룹번호 부여 결과")
    logger.log(f"원본 행 수: {len(df):,}")
    logger.log(f"결과 행 수: {len(df_merged):,} (모든 행 유지)")
    logger.log(f"STEP1 후보 그룹 수: {len(candidate_groups)}개")
    logger.log(f"STEP2 최종 중복 그룹 수: {len(final_groups)}개")
    logger.log(f"STEP1 총 그룹 수: {df_merged['그룹번호_STEP1'].nunique()}개")
    logger.log(f"STEP2 총 그룹 수: {df_merged['그룹번호_STEP2'].nunique()}개")

    if '_merged_from_indices' in df_merged.columns:
        df_merged = df_merged.drop(columns=['_merged_from_indices'])
        logger.log("_merged_from_indices 컬럼 제거됨")

    logger.section("5단계: 결과 저장")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f'korean_vendor_merged_hybrid_{timestamp}.xlsx')
    logger.log(f"결과 파일 저장 중: {output_file}")
    df_merged.to_excel(output_file, index=False)
    logger.log(f"저장 완료: {output_file}")
    logger.log("그룹번호로 필터링하여 같은 그룹의 업체들을 확인할 수 있습니다.")
else:
    logger.log("중복이 발견되지 않았습니다.")

total_time = time.time() - logger.start_time
logger.section("프로세스 완료")
logger.log(f"총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
logger.log("모든 작업이 완료되었습니다!")
print("\n" + "="*60)
print("처리 완료! (한국 업체용)")
print(f"로그 파일: {logger.log_file}")
print("="*60)
