"""
최종 최적화 버전: 다국어 중복 탐지 지원
- 다중 Blocking 키 생성 (같은 업체의 다른 언어 표기도 같은 그룹에 포함)
- 의미 기반 비교 (로마자화 + 문자열 비교)
- RapidFuzz 사용
"""
import pandas as pd
from typing import List, Dict, Set, Tuple
from rapidfuzz import fuzz
import numpy as np
import re
from collections import defaultdict

# 로마자화 라이브러리 (선택사항, 없어도 작동)
try:
    from pypinyin import lazy_pinyin
    PINYIN_AVAILABLE = True
except ImportError:
    PINYIN_AVAILABLE = False

try:
    import pykakasi
    KAKASI_AVAILABLE = True
except ImportError:
    KAKASI_AVAILABLE = False


class DuplicateDetectorFinal:
    """최종 최적화 버전: 다국어 중복 탐지 지원"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.duplicate_groups: List[List[int]] = []
        
        # 불용어 패턴 (다국어 지원)
        self.stopwords_patterns = [
            r'\b(주식회사|주\)|\(주\)|CORP|CORPORATION|CORP\.|INC|INC\.|INCORPORATED|LTD|LTD\.|LIMITED|CO|CO\.|COMPANY|COMP\.)',
            r'\b(LLC|LLC\.|LLP|LLP\.|PLC|PLC\.)',
            r'\b(有限会社|株式会社|有限会社|股份公司|有限公司|股份有限公司)',
            r'\b(S\.A\.|S\.A|SA|S\.R\.L\.|SRL|GMBH|GMBH\.|AG|AG\.)',
            r'\b(PT|PT\.|CV|CV\.|TBK|TBK\.)',
            r'\b(PTE|PTE\.|PVT|PVT\.|PRIVATE)',
        ]
        self.stopwords_regex = re.compile('|'.join(self.stopwords_patterns), re.IGNORECASE)
    
    def remove_stopwords(self, text: str) -> str:
        """불용어 제거"""
        if not text:
            return ""
        return self.stopwords_regex.sub('', text).strip()
    
    def normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        if pd.isna(text) or text is None:
            return ""
        text = str(text).strip()
        text = self.remove_stopwords(text)
        text = re.sub(r'[a-z]+', lambda m: m.group().upper(), text)
        text = ' '.join(text.split())
        return text
    
    def transliterate_chinese(self, text: str) -> str:
        """중국어 로마자화 (Pinyin)"""
        if not PINYIN_AVAILABLE:
            return ""
        try:
            chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
            if not chinese_chars:
                return ""
            pinyin_list = []
            for chars in chinese_chars:
                pinyin_list.extend(lazy_pinyin(chars))
            return ' '.join(pinyin_list).upper()
        except:
            return ""
    
    def transliterate_japanese(self, text: str) -> str:
        """일본어 로마자화 (Romaji)"""
        if not KAKASI_AVAILABLE:
            return ""
        try:
            japanese_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', text)
            if not japanese_chars:
                return ""
            kks = pykakasi.kakasi()
            result = kks.convert(''.join(japanese_chars))
            romaji = ' '.join([item['hepburn'] for item in result]).upper()
            return romaji
        except:
            return ""
    
    def extract_multilingual_prefixes(self, text: str, length: int = 5) -> List[str]:
        """다국어 prefix 추출: 원본 + 로마자화 버전"""
        prefixes = []
        
        if not text:
            return prefixes
        
        normalized = self.normalize_text(text)
        
        # 1. 원본 prefix (다국어 문자 포함)
        cleaned = re.sub(r'[^\uAC00-\uD7A3\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u00C0-\u1EF9a-zA-Z]', '', normalized)
        if cleaned:
            prefixes.append(cleaned[:length])
        
        # 2. 중국어 로마자화 prefix
        pinyin = self.transliterate_chinese(text)
        if pinyin:
            pinyin_clean = re.sub(r'[^a-zA-Z]', '', pinyin)
            if pinyin_clean:
                prefixes.append(pinyin_clean[:length])
        
        # 3. 일본어 로마자화 prefix
        romaji = self.transliterate_japanese(text)
        if romaji:
            romaji_clean = re.sub(r'[^a-zA-Z]', '', romaji)
            if romaji_clean:
                prefixes.append(romaji_clean[:length])
        
        # 4. 영어만 추출 (혼합된 경우)
        english_only = re.sub(r'[^a-zA-Z]', '', normalized)
        if english_only and english_only != cleaned[:length]:
            prefixes.append(english_only[:length])
        
        return list(set(prefixes))  # 중복 제거
    
    def extract_city(self, row: pd.Series) -> str:
        """주소에서 도시명 추출"""
        city_fields = ['CITY1', 'CITY2']
        for field in city_fields:
            if field in row.index and pd.notna(row[field]):
                city = self.normalize_text(str(row[field]))
                if city:
                    return city
        return ""
    
    def create_multiple_blocking_keys(self, row: pd.Series) -> List[str]:
        """
        다중 Blocking 키 생성 (핵심: 같은 업체의 다른 언어 표기도 같은 그룹에 포함)
        
        예:
        - "阿里巴巴集团" → ["CN|阿里巴巴|BEIJING", "CN|ALIBABA|BEIJING"]
        - "Alibaba Group" → ["CN|ALIBABA|BEIJING"]
        → 둘 다 "CN|ALIBABA|BEIJING" 키를 가지므로 같은 그룹에 포함됨!
        """
        keys = []
        
        # 국가 코드
        country = ""
        if 'Land' in row.index and pd.notna(row['Land']):
            country = str(row['Land']).strip().upper()
        
        # 도시명
        city = self.extract_city(row)
        
        # 업체명에서 다국어 prefix 추출
        name = row.get('공급업체명', '')
        if pd.notna(name) and name:
            name_prefixes = self.extract_multilingual_prefixes(str(name), length=5)
            
            if name_prefixes:
                # 각 prefix에 대해 키 생성
                for prefix in name_prefixes:
                    if prefix:
                        key_parts = [country, prefix]
                        if city:
                            key_parts.append(city)
                        keys.append('|'.join(key_parts))
            else:
                # prefix가 없으면 국가+도시만
                if country and city:
                    keys.append(f"{country}||{city}")
                elif country:
                    keys.append(f"{country}||")
        else:
            # 이름이 없으면 국가+도시만
            if country and city:
                keys.append(f"{country}||{city}")
            elif country:
                keys.append(f"{country}||")
        
        return keys if keys else ['UNKNOWN']
    
    def normalize_address(self, row: pd.Series) -> str:
        """주소 정규화"""
        address_parts = []
        address_fields = ['CITY1', 'CITY2', 'STREET', 'HOUSE_NUM1', 'HOUSE_NUM2', 
                         'STR_SUPPL1', 'STR_SUPPL2', 'STR_SUPPL3']
        
        for field in address_fields:
            if field in row.index and pd.notna(row[field]):
                part = self.normalize_text(str(row[field]))
                if part:
                    address_parts.append(part)
        
        return ' '.join(address_parts)
    
    def calculate_multilingual_similarity(self, name1: str, name2: str) -> float:
        """
        다국어 유사도 계산 (핵심 함수)
        - 원본 문자열 비교
        - 로마자화 버전 비교
        - 최대값 사용
        """
        name1_norm = self.normalize_text(name1)
        name2_norm = self.normalize_text(name2)
        
        if not name1_norm or not name2_norm:
            return 0.0
        
        # 1. 원본 문자열 유사도
        ratio1 = fuzz.ratio(name1_norm, name2_norm) / 100.0
        partial1 = fuzz.partial_ratio(name1_norm, name2_norm) / 100.0
        token_sort1 = fuzz.token_sort_ratio(name1_norm, name2_norm) / 100.0
        original_score = ratio1 * 0.4 + partial1 * 0.3 + token_sort1 * 0.3
        
        # 2. 로마자화 버전 비교
        pinyin1 = self.transliterate_chinese(name1)
        pinyin2 = self.transliterate_chinese(name2)
        romaji1 = self.transliterate_japanese(name1)
        romaji2 = self.transliterate_japanese(name2)
        
        romanized_scores = []
        
        # Pinyin 비교
        if pinyin1 and pinyin2:
            romanized_scores.append(fuzz.ratio(pinyin1, pinyin2) / 100.0)
            romanized_scores.append(fuzz.partial_ratio(pinyin1, pinyin2) / 100.0)
        
        # Romaji 비교
        if romaji1 and romaji2:
            romanized_scores.append(fuzz.ratio(romaji1, romaji2) / 100.0)
            romanized_scores.append(fuzz.partial_ratio(romaji1, romaji2) / 100.0)
        
        # 로마자화 vs 원본 비교 (다국어 매칭)
        if pinyin1:
            romanized_scores.append(fuzz.partial_ratio(pinyin1, name2_norm) / 100.0)
        if pinyin2:
            romanized_scores.append(fuzz.partial_ratio(name1_norm, pinyin2) / 100.0)
        if romaji1:
            romanized_scores.append(fuzz.partial_ratio(romaji1, name2_norm) / 100.0)
        if romaji2:
            romanized_scores.append(fuzz.partial_ratio(name1_norm, romaji2) / 100.0)
        
        # 최종 유사도: 원본과 로마자화 중 최대값
        max_romanized = max(romanized_scores) if romanized_scores else 0.0
        similarity = max(original_score, max_romanized * 0.9)
        
        return similarity
    
    def calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """주소 유사도"""
        addr1 = self.normalize_text(addr1)
        addr2 = self.normalize_text(addr2)
        
        if not addr1 or not addr2:
            return 0.0
        
        ratio = fuzz.ratio(addr1, addr2) / 100.0
        partial_ratio = fuzz.partial_ratio(addr1, addr2) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        
        similarity = (ratio * 0.3 + partial_ratio * 0.5 + token_sort_ratio * 0.2)
        return similarity
    
    def are_duplicates(self, row1: pd.Series, row2: pd.Series) -> Tuple[bool, float]:
        """중복 판단 (다국어 지원)"""
        name1 = row1.get('공급업체명', '')
        name2 = row2.get('공급업체명', '')
        
        # 다국어 유사도 계산
        name_similarity = self.calculate_multilingual_similarity(
            str(name1) if pd.notna(name1) else '',
            str(name2) if pd.notna(name2) else ''
        )
        
        addr1 = self.normalize_address(row1)
        addr2 = self.normalize_address(row2)
        address_similarity = self.calculate_address_similarity(addr1, addr2)
        
        land_match = False
        if 'Land' in row1.index and 'Land' in row2.index:
            land1 = self.normalize_text(str(row1['Land']))
            land2 = self.normalize_text(str(row2['Land']))
            land_match = (land1 == land2 and land1 != "")
        
        # 종합 판단
        if name_similarity >= self.similarity_threshold and address_similarity >= 0.7:
            confidence = (name_similarity * 0.6 + address_similarity * 0.4)
            return (True, confidence)
        
        if (not name1 or not name2) and address_similarity >= 0.85 and land_match:
            confidence = address_similarity * 0.9
            return (True, confidence)
        
        if name_similarity >= 0.9 and land_match:
            confidence = name_similarity * 0.8
            return (True, confidence)
        
        return (False, 0.0)
    
    def detect_duplicates(self, df: pd.DataFrame, show_progress: bool = True) -> List[List[int]]:
        """
        다국어 중복 탐지 (최적화 버전)
        
        핵심:
        1. 다중 Blocking 키 생성으로 같은 업체의 다른 언어 표기도 같은 그룹에 포함
        2. Blocking으로 비교 횟수 대폭 감소
        3. 다국어 유사도 계산으로 정확한 판단
        """
        n = len(df)
        visited = set()
        groups = []
        
        if show_progress:
            print("\n[1단계] 다국어 Blocking 키 생성 중...")
            if PINYIN_AVAILABLE:
                print("  ✓ 중국어 로마자화 지원: 활성화")
            else:
                print("  ⚠ 중국어 로마자화 지원: 비활성화 (pypinyin 설치 권장)")
            if KAKASI_AVAILABLE:
                print("  ✓ 일본어 로마자화 지원: 활성화")
            else:
                print("  ⚠ 일본어 로마자화 지원: 비활성화 (pykakasi 설치 권장)")
        
        # 다중 Blocking 키 생성
        blocking_groups = defaultdict(list)
        for idx in df.index:
            keys = self.create_multiple_blocking_keys(df.loc[idx])
            for key in keys:
                blocking_groups[key].append(idx)
        
        total_blocks = len(blocking_groups)
        if show_progress:
            print(f"  생성된 Blocking 그룹: {total_blocks}개")
            block_sizes = [len(indices) for indices in blocking_groups.values()]
            max_size = max(block_sizes) if block_sizes else 0
            avg_size = np.mean(block_sizes) if block_sizes else 0
            large_blocks = [s for s in block_sizes if s > 100]
            print(f"  최대 그룹 크기: {max_size}행, 평균: {avg_size:.1f}행")
            if large_blocks:
                print(f"  큰 그룹(100행 이상): {len(large_blocks)}개")
        
        # 각 Blocking 그룹 내에서 비교
        if show_progress:
            print("\n[2단계] 다국어 후보군 비교 시작...")
        
        block_num = 0
        total_comparisons = 0
        
        for blocking_key, indices in blocking_groups.items():
            block_num += 1
            block_size = len(indices)
            
            if block_size < 2:
                continue
            
            comparisons_in_block = block_size * (block_size - 1) // 2
            total_comparisons += comparisons_in_block
            
            if show_progress and block_num % 500 == 0:
                print(f"  처리 중: {block_num}/{total_blocks} 그룹, "
                      f"총 비교 횟수: {total_comparisons:,}")
            
            for i in indices:
                if i in visited:
                    continue
                
                current_group = [i]
                visited.add(i)
                
                for j in indices:
                    if j <= i or j in visited:
                        continue
                    
                    is_dup, confidence = self.are_duplicates(df.loc[i], df.loc[j])
                    
                    if is_dup:
                        current_group.append(j)
                        visited.add(j)
                        if show_progress and len(current_group) == 2:
                            name1 = str(df.loc[i].get('공급업체명', ''))[:40]
                            name2 = str(df.loc[j].get('공급업체명', ''))[:40]
                            print(f"  [중복 발견] {name1} <-> {name2} (신뢰도: {confidence:.2f})")
                
                if len(current_group) > 1:
                    groups.append(current_group)
        
        if show_progress:
            print(f"\n[완료] 총 비교 횟수: {total_comparisons:,}")
            if total_comparisons > 0:
                reduction = 151000000 / total_comparisons
                print(f"  (기존 방식 대비 약 {reduction:.0f}배 감소)")
        
        self.duplicate_groups = groups
        return groups
    
    def get_duplicate_groups(self) -> List[List[int]]:
        """탐지된 중복 그룹 반환"""
        return self.duplicate_groups

