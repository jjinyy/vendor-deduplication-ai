"""
하이브리드 버전: 다중 Blocking + 의미 기반 임베딩
- 1단계: 다중 Blocking 키로 후보군 추출 (빠름)
- 2단계: 후보군에 대해서만 의미 기반 비교 (정확)
"""
import pandas as pd
from typing import List, Dict, Set, Tuple
from rapidfuzz import fuzz
import numpy as np
import re
from collections import defaultdict

# 로마자화 라이브러리
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

# 의미 기반 임베딩 라이브러리
try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDING_AVAILABLE = True
    # GPU 사용 가능 여부 확인
    USE_GPU = torch.cuda.is_available()
except ImportError:
    EMBEDDING_AVAILABLE = False
    USE_GPU = False


class DuplicateDetectorHybrid:
    """하이브리드 버전: Blocking + 의미 기반 임베딩"""
    
    def __init__(self, similarity_threshold: float = 0.85, use_embedding: bool = True, logger=None):
        """
        Args:
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)
            use_embedding: 의미 기반 임베딩 사용 여부
            logger: 로거 객체 (선택사항)
        """
        self.similarity_threshold = similarity_threshold
        self.duplicate_groups: List[List[int]] = []
        self.use_embedding = use_embedding and EMBEDDING_AVAILABLE
        self.logger = logger
        
        # 의미 기반 임베딩 모델 로드
        if self.use_embedding:
            log_msg = "의미 기반 임베딩 모델 로딩 중..."
            if self.logger:
                self.logger.log(log_msg)
            else:
                print(log_msg)
            try:
                # multilingual 모델 사용
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                log_msg = f"  [OK] 모델 로드 완료 (GPU: {USE_GPU})"
                if self.logger:
                    self.logger.log(log_msg)
                else:
                    print(log_msg)
            except Exception as e:
                log_msg = f"  [WARNING] 모델 로드 실패: {e}"
                if self.logger:
                    self.logger.log(log_msg)
                    self.logger.log("  로마자화 방식으로 전환합니다.")
                else:
                    print(log_msg)
                    print("  로마자화 방식으로 전환합니다.")
                self.use_embedding = False
        
        # 불용어 패턴
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
        """중국어 로마자화"""
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
        """일본어 로마자화"""
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
        """다국어 prefix 추출"""
        prefixes = []
        if not text:
            return prefixes
        
        normalized = self.normalize_text(text)
        cleaned = re.sub(r'[^\uAC00-\uD7A3\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u00C0-\u1EF9a-zA-Z]', '', normalized)
        if cleaned:
            prefixes.append(cleaned[:length])
        
        pinyin = self.transliterate_chinese(text)
        if pinyin:
            pinyin_clean = re.sub(r'[^a-zA-Z]', '', pinyin)
            if pinyin_clean:
                prefixes.append(pinyin_clean[:length])
        
        romaji = self.transliterate_japanese(text)
        if romaji:
            romaji_clean = re.sub(r'[^a-zA-Z]', '', romaji)
            if romaji_clean:
                prefixes.append(romaji_clean[:length])
        
        english_only = re.sub(r'[^a-zA-Z]', '', normalized)
        if english_only and english_only != cleaned[:length]:
            prefixes.append(english_only[:length])
        
        return list(set(prefixes))
    
    def extract_city(self, row: pd.Series) -> str:
        """도시명 추출"""
        city_fields = ['CITY1', 'CITY2']
        for field in city_fields:
            if field in row.index and pd.notna(row[field]):
                city = self.normalize_text(str(row[field]))
                if city:
                    return city
        return ""
    
    def create_multiple_blocking_keys(self, row: pd.Series) -> List[str]:
        """다중 Blocking 키 생성"""
        keys = []
        country = ""
        if 'Land' in row.index and pd.notna(row['Land']):
            country = str(row['Land']).strip().upper()
        
        city = self.extract_city(row)
        name = row.get('공급업체명', '')
        
        if pd.notna(name) and name:
            name_prefixes = self.extract_multilingual_prefixes(str(name), length=5)
            if name_prefixes:
                for prefix in name_prefixes:
                    if prefix:
                        key_parts = [country, prefix]
                        if city:
                            key_parts.append(city)
                        keys.append('|'.join(key_parts))
            else:
                if country and city:
                    keys.append(f"{country}||{city}")
                elif country:
                    keys.append(f"{country}||")
        else:
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
    
    def calculate_semantic_similarity(self, name1: str, name2: str) -> float:
        """의미 기반 유사도 계산"""
        if not self.use_embedding:
            return 0.0
        
        try:
            # 임베딩 생성
            embeddings = self.embedding_model.encode([name1, name2], convert_to_numpy=True)
            
            # 코사인 유사도 계산
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            return 0.0
    
    def calculate_multilingual_similarity(self, name1: str, name2: str) -> float:
        """다국어 유사도 계산 (로마자화 + 의미 기반)"""
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
        if pinyin1 and pinyin2:
            romanized_scores.append(fuzz.ratio(pinyin1, pinyin2) / 100.0)
        if romaji1 and romaji2:
            romanized_scores.append(fuzz.ratio(romaji1, romaji2) / 100.0)
        if pinyin1:
            romanized_scores.append(fuzz.partial_ratio(pinyin1, name2_norm) / 100.0)
        if pinyin2:
            romanized_scores.append(fuzz.partial_ratio(name1_norm, pinyin2) / 100.0)
        
        max_romanized = max(romanized_scores) if romanized_scores else 0.0
        romanized_score = max(original_score, max_romanized * 0.9)
        
        # 3. 의미 기반 유사도 (하이브리드)
        semantic_score = 0.0
        if self.use_embedding:
            semantic_score = self.calculate_semantic_similarity(name1, name2)
        
        # 최종 유사도: 로마자화와 의미 기반 중 최대값
        if semantic_score > 0:
            # 의미 기반이 높으면 더 신뢰
            final_score = max(romanized_score, semantic_score * 1.1)  # 의미 기반에 가중치
        else:
            final_score = romanized_score
        
        return min(final_score, 1.0)  # 1.0 초과 방지
    
    def calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """주소 유사도"""
        addr1 = self.normalize_text(addr1)
        addr2 = self.normalize_text(addr2)
        if not addr1 or not addr2:
            return 0.0
        
        ratio = fuzz.ratio(addr1, addr2) / 100.0
        partial_ratio = fuzz.partial_ratio(addr1, addr2) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        return (ratio * 0.3 + partial_ratio * 0.5 + token_sort_ratio * 0.2)
    
    def are_duplicates(self, row1: pd.Series, row2: pd.Series) -> Tuple[bool, float]:
        """중복 판단"""
        name1 = row1.get('공급업체명', '')
        name2 = row2.get('공급업체명', '')
        
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
        """하이브리드 중복 탐지"""
        n = len(df)
        visited = set()
        groups = []
        
        if show_progress:
            log_msg = "\n[1단계] 다국어 Blocking 키 생성 중..."
            if self.logger:
                self.logger.log(log_msg)
            else:
                print(log_msg)
            
            status_msgs = []
            if self.use_embedding:
                status_msgs.append("  [OK] 의미 기반 임베딩: 활성화")
            else:
                status_msgs.append("  [INFO] 의미 기반 임베딩: 비활성화")
            if PINYIN_AVAILABLE:
                status_msgs.append("  [OK] 중국어 로마자화: 활성화")
            else:
                status_msgs.append("  [INFO] 중국어 로마자화: 비활성화")
            if KAKASI_AVAILABLE:
                status_msgs.append("  [OK] 일본어 로마자화: 활성화")
            else:
                status_msgs.append("  [INFO] 일본어 로마자화: 비활성화")
            
            for msg in status_msgs:
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)
        
        # Blocking 키 생성
        blocking_groups = defaultdict(list)
        for idx in df.index:
            keys = self.create_multiple_blocking_keys(df.loc[idx])
            for key in keys:
                blocking_groups[key].append(idx)
        
        total_blocks = len(blocking_groups)
        if show_progress:
            block_sizes = [len(indices) for indices in blocking_groups.values()]
            max_size = max(block_sizes) if block_sizes else 0
            msg1 = f"  생성된 Blocking 그룹: {total_blocks}개"
            msg2 = f"  최대 그룹 크기: {max_size}행"
            if self.logger:
                self.logger.log(msg1)
                self.logger.log(msg2)
            else:
                print(msg1)
                print(msg2)
        
        if show_progress:
            msg = "\n[2단계] 하이브리드 비교 시작 (Blocking + 의미 기반)..."
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
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
                msg = f"  처리 중: {block_num}/{total_blocks} 그룹, 총 비교 횟수: {total_comparisons:,}"
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)
            
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
                            msg = f"  [중복 발견] {name1} <-> {name2} (신뢰도: {confidence:.2f})"
                            if self.logger:
                                self.logger.log(msg)
                            else:
                                print(msg)
                
                if len(current_group) > 1:
                    groups.append(current_group)
        
        if show_progress:
            msg = f"\n[완료] 총 비교 횟수: {total_comparisons:,}"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        self.duplicate_groups = groups
        return groups
    
    def get_duplicate_groups(self) -> List[List[int]]:
        return self.duplicate_groups

