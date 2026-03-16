"""
Cheap Gate: Step2로 보내기 전 빠른 필터링
절대 고비용 연산(큰 레벤슈타인) 남발 금지
"""
import re
from typing import Dict, Set, Tuple


def get_script_key(text: str) -> str:
    """
    텍스트의 문자 스크립트 판별
    
    Returns:
        'han' (한자), 'kor' (한글), 'latin' (라틴), 'mixed', 'unknown'
    """
    if not text:
        return 'unknown'
    
    has_han = bool(re.search(r'[\u4e00-\u9fff]', text))
    has_kor = bool(re.search(r'[\uac00-\ud7a3]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    
    if has_han and not has_kor and not has_latin:
        return 'han'
    elif has_kor and not has_han and not has_latin:
        return 'kor'
    elif has_latin and not has_han and not has_kor:
        return 'latin'
    elif has_han or has_kor or has_latin:
        return 'mixed'
    else:
        return 'unknown'


def cheap_gate(i: int, j: int, row_meta_i: Dict, row_meta_j: Dict) -> Tuple[bool, str]:
    """
    Cheap Gate: 빠른 필터링
    
    Args:
        i, j: row id
        row_meta_i, row_meta_j: row 메타데이터
    
    Returns:
        (pass: bool, reason: str)
    """
    # 1. Country 불일치 체크 (UNK는 예외)
    country_i = row_meta_i.get('country_key', 'UNK')
    country_j = row_meta_j.get('country_key', 'UNK')
    
    if country_i != 'UNK' and country_j != 'UNK' and country_i != country_j:
        return (False, f"국가 불일치: {country_i} vs {country_j}")
    
    # 2. Script 불일치 체크 (너무 다르면 컷)
    script_i = row_meta_i.get('script_key', 'unknown')
    script_j = row_meta_j.get('script_key', 'unknown')
    
    # 한자 vs 라틴만 있는 경우는 허용 (다국어 매칭 가능)
    if script_i == 'han' and script_j == 'latin':
        pass  # 허용
    elif script_i == 'latin' and script_j == 'han':
        pass  # 허용
    elif script_i != script_j and script_i != 'mixed' and script_j != 'mixed':
        # 완전히 다른 스크립트면 컷 (단, mixed는 예외)
        return (False, f"스크립트 불일치: {script_i} vs {script_j}")
    
    # 3. Canonical key 토큰 overlap 체크
    tokens_i = row_meta_i.get('tokens', set())
    tokens_j = row_meta_j.get('tokens', set())
    
    if not tokens_i or not tokens_j:
        # 토큰이 없으면 일단 통과 (Step2에서 판단)
        return (True, "토큰 없음 - Step2로 위임")
    
    # 공통 토큰 계산
    common_tokens = tokens_i & tokens_j
    
    # 일반 단어만 공통인지 체크
    general_words = {
        'TRADING', 'COMMERCE', 'CO', 'LTD', 'LIMITED', 'INC', 'CORP', 'COMPANY',
        'PT', 'CV', 'TNHH', 'LTDA', 'ME', 'EIRELI', 'SA',
        '有限公司', '股份有限公司', '公司', '企业',
        '주식회사', '유한회사', '법인',
        'SEJAHTERA', 'MAKMUR', 'ABADI', 'JAYA', 'MANDIRI', 'PUTRA', 'SENTOSA',
        'CENTRAL', 'CENTRA', 'INDUSTRIAL', 'PART', 'INTERNATIONAL',
        'SOLUTION', 'SOLUTIONS', 'TECHNOLOGY', 'TECH', 'GROUP', 'HOLDINGS',
    }
    
    # 일반 단어만 공통이면 컷
    if common_tokens and common_tokens.issubset(general_words):
        return (False, f"일반 단어만 공통: {common_tokens}")
    
    # 고유 토큰 overlap 체크
    unique_tokens_i = tokens_i - general_words
    unique_tokens_j = tokens_j - general_words
    unique_common = unique_tokens_i & unique_tokens_j
    
    # 고유 토큰이 1개도 없으면 컷
    if not unique_common:
        return (False, "고유 토큰 공통 없음")
    
    # 4. 길이 비율 체크 (극단적으로 다르면 컷)
    len_i = row_meta_i.get('len', 0)
    len_j = row_meta_j.get('len', 0)
    
    if len_i > 0 and len_j > 0:
        len_ratio = min(len_i, len_j) / max(len_i, len_j)
        if len_ratio < 0.3:  # 길이가 3배 이상 차이나면 컷
            return (False, f"길이 비율 차이 큼: {len_i} vs {len_j} (ratio={len_ratio:.2f})")
    
    # 5. Prefix 간단 체크 (선택적)
    canonical_i = row_meta_i.get('canonical_key', '')
    canonical_j = row_meta_j.get('canonical_key', '')
    
    if canonical_i and canonical_j:
        # 첫 3글자라도 겹치면 통과
        prefix_i = canonical_i[:3].upper()
        prefix_j = canonical_j[:3].upper()
        
        if prefix_i and prefix_j and prefix_i == prefix_j:
            return (True, f"Prefix 일치: {prefix_i}")
    
    # 모든 체크 통과
    return (True, f"Cheap Gate 통과: 고유 토큰 {len(unique_common)}개 공통")


def compute_row_meta(row: Dict, name: str, country: str) -> Dict:
    """
    Row 메타데이터 계산
    
    Args:
        row: 원본 row dict
        name: 공급업체명
        country: 국가 코드
    
    Returns:
        row_meta 딕셔너리
    """
    import re
    
    # Canonical key: 불용어/법인형태 제거한 정규화된 이름
    # (기존 extract_company_core_name 로직 재사용 가능)
    canonical_key = name.upper().strip()
    
    # Embedding text: 의미 보존 (너무 공격적으로 정제하지 않음)
    embedding_text = name.strip()
    
    # Country key
    country_key = country.strip().upper() if country else 'UNK'
    
    # Script key
    script_key = get_script_key(name)
    
    # Tokens: cheap gate용 토큰 집합
    tokens = set(re.findall(r'\b\w+\b', name.upper()))
    
    # 길이
    len_norm = len(name)
    
    return {
        'canonical_key': canonical_key,
        'embedding_text': embedding_text,
        'country_key': country_key,
        'script_key': script_key,
        'tokens': tokens,
        'len': len_norm,
    }
