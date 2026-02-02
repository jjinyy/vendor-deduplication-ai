"""
최신 로그 기반 예상 시간 계산
"""
import re
from datetime import datetime, timedelta

log_file = 'process_log_hybrid.txt'

# 로그 파싱
with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# 정보 추출
total_groups = None
current_group = None
current_group_size = None
max_group_size = None
start_time = None
last_time = None
large_group_current = None
large_group_total = None

for line in lines:
    if '프로세스 시작' in line:
        match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        if match:
            start_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
    
    if '생성된 Blocking 그룹:' in line:
        match = re.search(r'생성된 Blocking 그룹:\s+([\d,]+)개', line)
        if match:
            total_groups = int(match.group(1).replace(',', ''))
    
    if '최대 그룹 크기:' in line:
        match = re.search(r'최대 그룹 크기:\s+(\d+)행', line)
        if match:
            max_group_size = int(match.group(1))
    
    if '처리 중:' in line:
        match = re.search(r'처리 중:\s+(\d+)/(\d+)\s+그룹', line)
        if match:
            current_group = int(match.group(1))
            total_groups = int(match.group(2))
        
        match = re.search(r'현재 그룹 크기:\s+(\d+)행', line)
        if match:
            current_group_size = int(match.group(1))
        
        match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        if match:
            last_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
    
    if '큰 그룹 진행:' in line:
        match = re.search(r'큰 그룹 진행:\s+(\d+)/(\d+)행', line)
        if match:
            large_group_current = int(match.group(1))
            large_group_total = int(match.group(2))
        
        match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        if match:
            last_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')

# 큰 그룹 진행 시간 계산
large_group_times = []
prev_time = None
prev_progress = None

for line in lines:
    if '큰 그룹 진행:' in line:
        match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        match2 = re.search(r'큰 그룹 진행:\s+(\d+)/(\d+)행', line)
        if match and match2:
            current_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
            current_progress = int(match2.group(1))
            
            if prev_time and prev_progress:
                elapsed = (current_time - prev_time).total_seconds()
                rows_processed = current_progress - prev_progress
                if rows_processed > 0:
                    large_group_times.append((rows_processed, elapsed))
            
            prev_time = current_time
            prev_progress = current_progress

# 결과 출력
print("="*80)
print("예상 완료 시간 계산 (최신 로그 기반)")
print("="*80)
print()

if total_groups:
    print(f"전체 Blocking 그룹: {total_groups:,}개")
if current_group:
    print(f"현재 처리 중: {current_group:,}/{total_groups:,} 그룹 ({current_group/total_groups*100:.2f}%)")
if current_group_size:
    print(f"현재 그룹 크기: {current_group_size:,}행")
if large_group_current and large_group_total:
    print(f"큰 그룹 진행: {large_group_current:,}/{large_group_total:,}행 ({large_group_current/large_group_total*100:.2f}%)")
if max_group_size:
    print(f"최대 그룹 크기: {max_group_size:,}행")
print()

# 현재 큰 그룹 남은 시간
if large_group_times and large_group_current and large_group_total:
    recent_times = large_group_times[-5:] if len(large_group_times) >= 5 else large_group_times
    total_rows = sum(t[0] for t in recent_times)
    total_time = sum(t[1] for t in recent_times)
    
    if total_time > 0:
        seconds_per_row = total_time / total_rows
        remaining_rows = large_group_total - large_group_current
        remaining_seconds = remaining_rows * seconds_per_row
        remaining_time = timedelta(seconds=int(remaining_seconds))
        
        print(f"현재 큰 그룹:")
        print(f"  - 남은 행: {remaining_rows:,}행")
        print(f"  - 예상 남은 시간: {remaining_time}")
        print()

# 전체 예상 시간 계산
if start_time and last_time and current_group and total_groups:
    elapsed = (last_time - start_time).total_seconds()
    
    if current_group > 0:
        # 평균 그룹당 시간
        time_per_group = elapsed / current_group
        
        # 남은 그룹 수
        remaining_groups = total_groups - current_group
        
        # 예상 남은 시간
        remaining_seconds = remaining_groups * time_per_group
        
        # 현재 큰 그룹 남은 시간 추가
        if large_group_current and large_group_total and large_group_times:
            recent_times = large_group_times[-5:] if len(large_group_times) >= 5 else large_group_times
            total_rows = sum(t[0] for t in recent_times)
            total_time = sum(t[1] for t in recent_times)
            if total_time > 0:
                seconds_per_row = total_time / total_rows
                remaining_rows = large_group_total - large_group_current
                large_group_remaining_time = remaining_rows * seconds_per_row
                # 큰 그룹 남은 시간을 전체 남은 시간에 추가
                remaining_seconds += large_group_remaining_time
        
        remaining_time = timedelta(seconds=int(remaining_seconds))
        estimated_completion = last_time + remaining_time
        
        print(f"전체 예상 시간:")
        print(f"  - 경과 시간: {timedelta(seconds=int(elapsed))}")
        print(f"  - 평균 그룹당 시간: {time_per_group:.2f}초")
        print(f"  - 남은 그룹: {remaining_groups:,}개")
        print(f"  - 예상 남은 시간: {remaining_time}")
        print(f"  - 예상 완료 시간: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 시간을 더 읽기 쉬운 형식으로 변환
        hours = remaining_seconds / 3600
        days = hours / 24
        
        if days >= 1:
            print(f"  → 약 {days:.1f}일 ({hours:.1f}시간)")
        elif hours >= 1:
            print(f"  → 약 {hours:.1f}시간 ({remaining_seconds/60:.0f}분)")
        else:
            print(f"  → 약 {remaining_seconds/60:.0f}분")

print("="*80)

