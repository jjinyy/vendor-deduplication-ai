"""
하이브리드 버전 진행 상황 확인
"""
import os
from datetime import datetime

log_file = 'process_log_hybrid.txt'

if os.path.exists(log_file):
    print("="*60)
    print("하이브리드 버전 진행 상황")
    print("="*60)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(''.join(lines[-30:]))
    
    mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
    time_since_update = (datetime.now() - mtime).total_seconds()
    print(f"\n마지막 업데이트: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"업데이트 후 경과: {time_since_update/60:.1f}분")
    
    # 결과 파일 확인
    result_files = [f for f in os.listdir('.') if 'hybrid' in f and f.endswith('.xlsx')]
    if result_files:
        print(f"\n생성된 결과 파일:")
        for f in sorted(result_files)[-3:]:
            size = os.path.getsize(f)
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"  - {f} ({size:,} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
else:
    print("로그 파일이 아직 생성되지 않았습니다.")

