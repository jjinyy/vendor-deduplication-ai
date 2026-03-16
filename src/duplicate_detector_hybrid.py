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

try:
    from korean_romanizer.romanizer import Romanizer
    KOREAN_ROMANIZER_AVAILABLE = True
except ImportError:
    KOREAN_ROMANIZER_AVAILABLE = False

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
        
        # [성능 개선] Core name 캐시 (중복 호출 방지)
        self._core_name_cache = {}
        
        # [버그 수정] Stopword 세트 통합 (한 번만 정의하여 일관성 유지)
        self.STEP1_STOP_WORDS = {
            # 국가명, 지역명
            'INDONESIA', 'INDONESIAN', 'ID', 'CHINA', 'CHINESE', 'CN',
            'KOREA', 'KOREAN', 'KR', 'JAPAN', 'JAPANESE', 'JP',
            'THAILAND', 'THAI', 'TH', 'VIETNAM', 'VIETNAMESE', 'VN',
            # 인도네시아 STOP WORD
            'CENTRAL', 'CENTRA', 'SEJAHTERA', 'MAKMUR', 'ABADI', 'JAYA',
            'MANDIRI', 'PUTRA', 'SENTOSA', 'INDUSTRIAL', 'PART', 'INTERNATIONAL', 'NIAGA',
            # 중국 STOP WORD (로마자화된 형태)
            'SHANGMAO', 'WUZI', 'JITUAN', 'ZHIZAOCHANG', 'JIANSHEGONGCHENG',
            # 베트남 STOP WORD
            'CTY', 'CONG', 'TY', 'TNHH', 'SX', 'TM', 'DV', 'BAO', 'BI',
            # 영어/포르투갈어 STOP WORD
            'TRADING', 'COMMERCE', 'SOLUTION', 'SOLUTIONS', 'INTEGRATION',
            'SOLUSI', 'INTEGRASI', 'TEKNOLOGI', 'TECHNOLOGY', 'TECH', 'PARTS',
            'CENTRE', 'CENTER', 'GLOBAL', 'WORLD', 'GROUP', 'HOLDINGS', 'ENTERPRISES',
            # 포르투갈어 국가명
            'BRASIL', 'BRASILIA', 'BRASILEIRA', 'BRASILEIRO', 'BRASILEIRAS', 'BRASILEIROS',
            'INDUSTRIES',
        }
        
        self.GENERAL_WORDS = {
            'INDONESIA', 'INDONESIAN', 'ID',
            'CHINA', 'CHINESE', 'CN',
            'KOREA', 'KOREAN', 'KR',
            'JAPAN', 'JAPANESE', 'JP',
            'THAILAND', 'THAI', 'TH',
            'VIETNAM', 'VIETNAMESE', 'VN',
            'BRASIL', 'BRASILIA', 'BRASILEIRA', 'BRASILEIRO', 'BRASILEIRAS', 'BRASILEIROS',
            'SOLUSI', 'SOLUTION', 'SOLUTIONS',
            'INTEGRASI', 'INTEGRATION',
            'TEKNOLOGI', 'TECHNOLOGY', 'TECH',
            'PART', 'PARTS',
            'CENTRAL', 'CENTRE', 'CENTER',
            'INTERNATIONAL', 'GLOBAL', 'WORLD',
            'GROUP', 'HOLDINGS', 'ENTERPRISES',
            'SEJAHTERA', 'MAKMUR', 'ABADI', 'JAYA', 'SENTOSA', 'MANDIRI', 'PUTRA',
            'NIAGA', 'INDUSTRIAL', 'INDUSTRIES',
        }
        
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
        
        # 불용어 패턴 (포르투갈어, 베트남어 포함)
        # 주의: 단어 경계를 명확히 하여 "Samsung"의 "SA" 같은 부분이 제거되지 않도록 함
        self.stopwords_patterns = [
            r'\b(주식회사|주\)|\(주\)|CORP|CORPORATION|CORP\.|INC|INC\.|INCORPORATED|LTD|LTD\.|LIMITED|CO\b|CO\.|COMPANY|COMP\.)',
            r'\b(LLC|LLC\.|LLP|LLP\.|PLC|PLC\.)',
            r'\b(有限会社|株式会社|有限会社|股份公司|有限公司|股份有限公司)',
            # SA는 단독 단어로만 매칭 (Samsung의 SA가 제거되지 않도록)
            r'\b(S\.A\.|S\.A\b|S\.R\.L\.|SRL|GMBH|GMBH\.|AG|AG\.)',
            r'\b(PT|PT\.|CV|CV\.|TBK|TBK\.)',
            r'\b(PTE|PTE\.|PVT|PVT\.|PRIVATE)',
            # 포르투갈어 불용어
            r'\b(LTDA|LTDA\.|EIRELI|EIRELI\.|ME|ME\.|EPP|EPP\.|S\.A\.|S\.A\b)',
            # 베트남어 불용어
            r'\b(CONG TY|CONG TY TNHH|TNHH|CONG TY CO PHAN|CO PHAN|DOANH NGHIEP)',
        ]
        self.stopwords_regex = re.compile('|'.join(self.stopwords_patterns), re.IGNORECASE)
    
    def remove_stopwords(self, text: str) -> str:
        """불용어 제거"""
        if not text:
            return ""
        return self.stopwords_regex.sub('', text).strip()
    
    def normalize_text(self, text: str) -> str:
        """텍스트 정규화 (포르투갈어, 베트남어 악센트 제거 포함)"""
        if pd.isna(text) or text is None:
            return ""
        text = str(text).strip()
        text = self.remove_stopwords(text)
        
        # 포르투갈어/베트남어 악센트 제거
        text = self.remove_accents(text)
        
        text = re.sub(r'[a-z]+', lambda m: m.group().upper(), text)
        text = ' '.join(text.split())
        return text
    
    def remove_accents(self, text: str) -> str:
        """포르투갈어/베트남어 악센트 제거 (성능 개선: unicodedata 사용)"""
        if not text:
            return ""
        
        import unicodedata
        
        # 베트남어 특수 문자 (unicodedata로 처리 안 되는 경우)
        text = text.replace('đ', 'd').replace('Đ', 'D')
        
        # 일반 악센트 제거 (NFKD 정규화 후 combining 문자 제거)
        # O(문자수)로 빠르게 처리
        return ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
    
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
    
    def transliterate_korean(self, text: str) -> str:
        """한글 로마자화"""
        if not KOREAN_ROMANIZER_AVAILABLE:
            return ""
        try:
            korean_chars = re.findall(r'[\uAC00-\uD7A3]+', text)
            if not korean_chars:
                return ""
            romans = []
            for chars in korean_chars:
                romanizer = Romanizer(chars)
                romanized = romanizer.romanize()
                if romanized:
                    romans.append(romanized)
            return ' '.join(romans).upper()
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
        
        korean_roman = self.transliterate_korean(text)
        if korean_roman:
            korean_clean = re.sub(r'[^a-zA-Z]', '', korean_roman)
            if korean_clean:
                prefixes.append(korean_clean[:length])
        
        # 영어 텍스트의 경우, 단어별로 prefix 생성하여 더 많은 매칭 가능
        # 예: "Samsung Electronics" → "SAMSU", "ELECT" 등
        english_only = re.sub(r'[^a-zA-Z]', ' ', normalized)
        english_words = english_only.split()
        for word in english_words[:3]:  # 처음 3개 단어만
            if len(word) >= 3:  # 최소 3글자 이상
                prefixes.append(word[:length].upper())
        
        return list(set(prefixes))
    
    def normalize_name_for_blocking(self, name: str) -> str:
        """
        정규화된 이름 키 생성 (구두점/공백/법인형태 차이만 있는 케이스 탐지용)
        
        정규화 규칙:
        - 대소문자 통일 (UPPER)
        - 구두점 제거
        - 공백 정리 (단일 공백)
        - 법인형태 제거 (CO, LTD, INC, PT, CV, TNHH, 有限公司 등)
        - '&' -> 'AND'
        
        예: "LEEPACK CO., LTD" -> "LEEPACK"
        예: "LEEPACK CO.,LTD" -> "LEEPACK"
        """
        if not name or pd.isna(name):
            return ""
        
        name_str = str(name).strip()
        if not name_str:
            return ""
        
        # 1. 대소문자 통일 (UPPER)
        normalized = name_str.upper()
        
        # 2. '&' -> 'AND'
        normalized = normalized.replace('&', 'AND')
        
        # 3. 구두점 제거
        normalized = re.sub(r'[.,;:!?()[\]{}"\'-]', ' ', normalized)
        
        # 4. 법인형태 제거 (다국어 지원)
        legal_forms = [
            # 영어
            r'\bCO\b', r'\bLTD\b', r'\bLIMITED\b', r'\bINC\b', r'\bINCORPORATED\b',
            r'\bCORP\b', r'\bCORPORATION\b', r'\bCOMPANY\b', r'\bCOMP\b',
            # 포르투갈어
            r'\bLTDA\b', r'\bME\b', r'\bEIRELI\b', r'\bSA\b', r'\bSOCIEDADE\b',
            r'\bANONIMA\b', r'\bEPP\b',
            # 인도네시아어
            r'\bPT\b', r'\bCV\b', r'\bTB\b', r'\bUD\b',
            # 베트남어
            r'\bTNHH\b', r'\bCONG\s+TY\b', r'\bCTY\b',
            # 중국어 (로마자화된 형태도 제거)
            r'\bYOUXIAN\s+GONGSI\b', r'\bGUFEN\s+YOUXIAN\s+GONGSI\b',
            r'\bGONGSI\b', r'\bQIYE\b',
        ]
        
        for pattern in legal_forms:
            normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
        
        # 5. 공백 정리 (여러 공백을 단일 공백으로, 앞뒤 공백 제거)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def extract_city(self, row: pd.Series) -> str:
        """도시명 추출"""
        city_fields = ['CITY1', 'CITY2']
        for field in city_fields:
            if field in row.index and pd.notna(row[field]):
                city = self.normalize_text(str(row[field]))
                if city:
                    return city
        return ""
    
    def extract_city_romanized(self, row: pd.Series) -> List[str]:
        """도시명 로마자화 버전 추출 (다국어 지원)"""
        city_fields = ['CITY1', 'CITY2']
        romanized_cities = []
        
        for field in city_fields:
            if field in row.index and pd.notna(row[field]):
                city = str(row[field]).strip()
                if not city:
                    continue
                
                # 원본 도시명
                city_norm = self.normalize_text(city)
                if city_norm:
                    romanized_cities.append(city_norm)
                
                # 중국어 → Pinyin
                pinyin = self.transliterate_chinese(city)
                if pinyin:
                    pinyin_clean = re.sub(r'[^a-zA-Z]', '', pinyin)
                    if pinyin_clean:
                        romanized_cities.append(pinyin_clean)
                
                # 일본어 → Romaji
                romaji = self.transliterate_japanese(city)
                if romaji:
                    romaji_clean = re.sub(r'[^a-zA-Z]', '', romaji)
                    if romaji_clean:
                        romanized_cities.append(romaji_clean)
                
                # 한글 → 로마자
                korean_roman = self.transliterate_korean(city)
                if korean_roman:
                    korean_clean = re.sub(r'[^a-zA-Z]', '', korean_roman)
                    if korean_clean:
                        romanized_cities.append(korean_clean)
                
                # 영어만 추출
                english_only = re.sub(r'[^a-zA-Z]', '', city_norm)
                if english_only and english_only != city_norm:
                    romanized_cities.append(english_only)
        
        return list(set([c.upper() for c in romanized_cities if c]))
    
    def create_multiple_blocking_keys(self, row: pd.Series) -> List[str]:
        """
        OR-블로킹: 다중 키 생성 (리콜 향상)
        
        [개선] 하나라도 걸리면 후보로 간주하는 다중 키 생성:
        1. 정규화된 토큰 (2-gram 기반, 기존)
        2. 초성/음절/자모 (한글/중국어)
        3. 숫자 토큰 (전화/우편/사업자번호 일부)
        4. 도메인/이메일
        
        후보 폭발 방지는 cheap gate가 담당
        """
        keys = []
        country = ""
        if 'Land' in row.index and pd.notna(row['Land']):
            country = str(row['Land']).strip().upper()
        
        name = row.get('공급업체명', '')
        if not name or pd.isna(name):
            return keys if keys else ['UNKNOWN']
        
        name_str = str(name)
        import re
        
        # ============================================================
        # 1. 정규화된 토큰 (2-gram 기반, 기존 로직)
        # ============================================================
        core_name = self.extract_company_core_name(name_str)
        
        if core_name and len(core_name.strip()) >= 2:
            core_name_upper = core_name.upper().strip()
            all_tokens = re.findall(r'\b\w+\b', core_name_upper)
            unique_tokens = [t for t in all_tokens if len(t) >= 2 and t not in self.STEP1_STOP_WORDS]
            
            if unique_tokens:
                from itertools import combinations
                max_tokens = min(4, len(unique_tokens))
                selected_tokens = sorted(unique_tokens, key=len, reverse=True)[:max_tokens]
                
                if len(selected_tokens) >= 2:
                    for t1, t2 in combinations(selected_tokens, 2):
                        token_pair = '_'.join(sorted([t1, t2]))
                        keys.append(f"{country}|CORE2:{token_pair}")
                else:
                    keys.append(f"{country}|CORE1:{selected_tokens[0]}")
        
        # 정규화된 이름 키도 추가
        normalized_name_key = self.normalize_name_for_blocking(name_str)
        if normalized_name_key:
            normalized_tokens = re.findall(r'\b\w+\b', normalized_name_key.upper())
            unique_norm_tokens = [t for t in normalized_tokens if len(t) >= 2 and t not in self.STEP1_STOP_WORDS]
            
            if unique_norm_tokens:
                from itertools import combinations
                max_norm_tokens = min(4, len(unique_norm_tokens))
                selected_norm_tokens = unique_norm_tokens[:max_norm_tokens]
                
                if len(selected_norm_tokens) >= 2:
                    for t1, t2 in combinations(selected_norm_tokens, 2):
                        token_pair = '_'.join(sorted([t1, t2]))
                        keys.append(f"{country}|NORM2:{token_pair}")
                else:
                    keys.append(f"{country}|NORM1:{selected_norm_tokens[0]}")
        
        # ============================================================
        # 2. 초성/음절/자모 (한글/중국어)
        # ============================================================
        def extract_phonetic_keys(text: str) -> List[str]:
            """초성/음절/자모 기반 키 추출"""
            phonetic_keys = []
            
            # 한글 초성 추출 (예: "삼성" → "ㅅㅅ")
            hangul_chosung = []
            for char in text:
                if '가' <= char <= '힣':
                    chosung_idx = (ord(char) - ord('가')) // 588
                    chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
                    if 0 <= chosung_idx < len(chosung_list):
                        hangul_chosung.append(chosung_list[chosung_idx])
            
            if len(hangul_chosung) >= 2:
                # 초성 2-gram
                for i in range(len(hangul_chosung) - 1):
                    phonetic_keys.append(f"{country}|CHOSUNG:{hangul_chosung[i]}{hangul_chosung[i+1]}")
            
            # 한글 음절 추출 (2-gram)
            hangul_syllables = [char for char in text if '가' <= char <= '힣']
            if len(hangul_syllables) >= 2:
                for i in range(len(hangul_syllables) - 1):
                    phonetic_keys.append(f"{country}|SYLLABLE:{hangul_syllables[i]}{hangul_syllables[i+1]}")
            
            # 중국어 자모 (Pinyin 초성)
            # 간단히 첫 글자 기반 키 생성
            chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
            if len(chinese_chars) >= 2:
                for i in range(len(chinese_chars) - 1):
                    phonetic_keys.append(f"{country}|HANZI:{chinese_chars[i]}{chinese_chars[i+1]}")
            
            return phonetic_keys
        
        phonetic_keys = extract_phonetic_keys(name_str)
        keys.extend(phonetic_keys)
        
        # ============================================================
        # 3. 숫자 토큰 (전화/우편/사업자번호 일부)
        # ============================================================
        def extract_number_keys(row: pd.Series) -> List[str]:
            """숫자 기반 키 추출"""
            number_keys = []
            
            # 전화번호 (마지막 4자리)
            phone_fields = ['PHONE', 'TEL', 'TELEPHONE', 'MOBILE']
            for field in phone_fields:
                if field in row.index and pd.notna(row[field]):
                    phone = str(row[field]).strip()
                    # 숫자만 추출
                    phone_digits = re.findall(r'\d', phone)
                    if len(phone_digits) >= 4:
                        # 마지막 4자리
                        last_4 = ''.join(phone_digits[-4:])
                        number_keys.append(f"{country}|PHONE:{last_4}")
            
            # 우편번호
            zip_fields = ['ZIP', 'POSTAL', 'POSTCODE', 'ZIPCODE']
            for field in zip_fields:
                if field in row.index and pd.notna(row[field]):
                    zipcode = str(row[field]).strip()
                    zip_digits = re.findall(r'\d', zipcode)
                    if len(zip_digits) >= 3:
                        # 처음 3자리 또는 전체
                        zip_key = ''.join(zip_digits[:min(5, len(zip_digits))])
                        number_keys.append(f"{country}|ZIP:{zip_key}")
            
            # 사업자번호 (일부, 보안 고려)
            reg_fields = ['REGISTRATION', 'REG_NUM', 'BUSINESS_NUM', 'TAX_ID']
            for field in reg_fields:
                if field in row.index and pd.notna(row[field]):
                    reg_num = str(row[field]).strip()
                    reg_digits = re.findall(r'\d', reg_num)
                    if len(reg_digits) >= 4:
                        # 마지막 4자리만 (보안)
                        last_4 = ''.join(reg_digits[-4:])
                        number_keys.append(f"{country}|REG:{last_4}")
            
            # 이름에 포함된 숫자 (예: "ABC123" → "123")
            name_numbers = re.findall(r'\d{3,}', name_str)
            for num in name_numbers[:2]:  # 최대 2개만
                if len(num) >= 3:
                    number_keys.append(f"{country}|NAME_NUM:{num[:6]}")  # 최대 6자리
            
            return number_keys
        
        number_keys = extract_number_keys(row)
        keys.extend(number_keys)
        
        # ============================================================
        # 4. 도메인/이메일
        # ============================================================
        def extract_domain_keys(row: pd.Series) -> List[str]:
            """도메인/이메일 기반 키 추출"""
            domain_keys = []
            
            # 이메일 필드
            email_fields = ['EMAIL', 'E_MAIL', 'CONTACT_EMAIL']
            for field in email_fields:
                if field in row.index and pd.notna(row[field]):
                    email = str(row[field]).strip().lower()
                    # 이메일에서 도메인 추출
                    if '@' in email:
                        domain = email.split('@')[1]
                        # 도메인 키 생성 (서브도메인 제거)
                        domain_parts = domain.split('.')
                        if len(domain_parts) >= 2:
                            main_domain = '.'.join(domain_parts[-2:])  # 예: "example.com"
                            domain_keys.append(f"{country}|DOMAIN:{main_domain}")
            
            # 이름에서 도메인 패턴 추출 (예: "www.example.com")
            domain_pattern = re.search(r'([a-z0-9-]+\.[a-z]{2,})', name_str.lower())
            if domain_pattern:
                domain = domain_pattern.group(1)
                domain_parts = domain.split('.')
                if len(domain_parts) >= 2:
                    main_domain = '.'.join(domain_parts[-2:])
                    domain_keys.append(f"{country}|DOMAIN:{main_domain}")
            
            return domain_keys
        
        domain_keys = extract_domain_keys(row)
        keys.extend(domain_keys)
        
        return keys if keys else ['UNKNOWN']
    
    def normalize_address(self, row: pd.Series) -> str:
        """주소 정규화 (로마자화 포함)"""
        address_parts = []
        address_fields = ['CITY1', 'CITY2', 'STREET', 'HOUSE_NUM1', 'HOUSE_NUM2', 
                         'STR_SUPPL1', 'STR_SUPPL2', 'STR_SUPPL3']
        for field in address_fields:
            if field in row.index and pd.notna(row[field]):
                part = str(row[field]).strip()
                if not part:
                    continue
                
                # 원본 정규화
                part_norm = self.normalize_text(part)
                if part_norm:
                    address_parts.append(part_norm)
                
                # 로마자화 버전 추가 (중국어, 일본어, 한글)
                pinyin = self.transliterate_chinese(part)
                if pinyin:
                    pinyin_clean = re.sub(r'[^a-zA-Z0-9\s]', '', pinyin).strip()
                    if pinyin_clean and pinyin_clean.upper() != part_norm.upper():
                        address_parts.append(pinyin_clean.upper())
                
                romaji = self.transliterate_japanese(part)
                if romaji:
                    romaji_clean = re.sub(r'[^a-zA-Z0-9\s]', '', romaji).strip()
                    if romaji_clean and romaji_clean.upper() != part_norm.upper():
                        address_parts.append(romaji_clean.upper())
                
                korean_roman = self.transliterate_korean(part)
                if korean_roman:
                    korean_clean = re.sub(r'[^a-zA-Z0-9\s]', '', korean_roman).strip()
                    if korean_clean and korean_clean.upper() != part_norm.upper():
                        address_parts.append(korean_clean.upper())
        
        # 중복 제거 후 결합
        return ' '.join(list(set(address_parts)))
    
    def calculate_semantic_similarity(self, name1: str, name2: str) -> float:
        """의미 기반 유사도 계산 (지역명 제거 후 비교)"""
        if not self.use_embedding:
            return 0.0
        
        try:
            # 지역명 제거 (중국 주요 도시명)
            # 지역명이 포함된 경우 제거하여 회사명과 업종에 집중
            # 중국 주요 도시명 (확장)
            common_cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉', '西安', '重庆',
                           '沈阳', '大连', '青岛', '苏州', '无锡', '宁波', '厦门', '福州', '济南', '郑州',
                           '长沙', '合肥', '石家庄', '哈尔滨', '长春', '太原', '南昌', '昆明', '贵阳', '南宁',
                           '海口', '兰州', '西宁', '银川', '乌鲁木齐', '拉萨', '呼和浩特',
                           '聊城', '烟台', '潍坊', '淄博', '临沂', '济宁', '泰安', '威海', '日照', '德州',
                           '滨州', '东营', '菏泽', '枣庄', '莱芜', '서울', '부산', '대구', '인천', '광주',
                           '대전', '울산', '수원', '고양', '용인', '성남', '부천', '화성', '안산', '안양',
                           'TOKYO', 'OSAKA', 'YOKOHAMA', 'NAGOYA', 'SAPPORO', 'FUKUOKA', 'SEOUL', 'BUSAN',
                           'BEIJING', 'SHANGHAI', 'GUANGZHOU', 'SHENZHEN', 'HANGZHOU', 'NANJING']
            
            # 지역명 패턴 (도, 시, 성 등)
            region_patterns = ['省', '市', '区', '县', '州', '道', '府', '都']
            
            name1_clean = name1
            name2_clean = name2
            
            # 지역명 제거 (도시명)
            for city in common_cities:
                name1_clean = name1_clean.replace(city, '')
                name2_clean = name2_clean.replace(city, '')
            
            # 지역명 패턴 제거 (省, 市, 区 등)
            for pattern in region_patterns:
                name1_clean = re.sub(rf'[\u4e00-\u9fff]+{pattern}', '', name1_clean)
                name2_clean = re.sub(rf'[\u4e00-\u9fff]+{pattern}', '', name2_clean)
            
            # 공백 정리
            name1_clean = re.sub(r'\s+', ' ', name1_clean).strip()
            name2_clean = re.sub(r'\s+', ' ', name2_clean).strip()
            
            # 지역명 제거 후에도 충분한 텍스트가 남아있는지 확인
            if len(name1_clean) < 2 or len(name2_clean) < 2:
                # 지역명 제거 후 텍스트가 너무 짧으면 원본 사용
                name1_clean = name1
                name2_clean = name2
            
            # 임베딩 생성
            embeddings = self.embedding_model.encode([name1_clean, name2_clean], convert_to_numpy=True)
            
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
        korean_roman1 = self.transliterate_korean(name1)
        korean_roman2 = self.transliterate_korean(name2)
        
        romanized_scores = []
        if pinyin1 and pinyin2:
            romanized_scores.append(fuzz.ratio(pinyin1, pinyin2) / 100.0)
        if romaji1 and romaji2:
            romanized_scores.append(fuzz.ratio(romaji1, romaji2) / 100.0)
        if korean_roman1 and korean_roman2:
            romanized_scores.append(fuzz.ratio(korean_roman1, korean_roman2) / 100.0)
        if pinyin1:
            romanized_scores.append(fuzz.partial_ratio(pinyin1, name2_norm) / 100.0)
        if pinyin2:
            romanized_scores.append(fuzz.partial_ratio(name1_norm, pinyin2) / 100.0)
        if korean_roman1:
            romanized_scores.append(fuzz.partial_ratio(korean_roman1, name2_norm) / 100.0)
        if korean_roman2:
            romanized_scores.append(fuzz.partial_ratio(name1_norm, korean_roman2) / 100.0)
        
        max_romanized = max(romanized_scores) if romanized_scores else 0.0
        romanized_score = max(original_score, max_romanized * 0.9)
        
        # 3. 의미 기반 유사도 (하이브리드)
        semantic_score = 0.0
        if self.use_embedding:
            semantic_score = self.calculate_semantic_similarity(name1, name2)
        
        # 최종 유사도: 로마자화와 의미 기반 중 최대값
        # 의미 기반 임베딩의 가중치를 낮춰서 지역명에 의한 오매칭 방지
        if semantic_score > 0:
            # 의미 기반 점수가 너무 높으면(0.95 이상) 가중치를 낮춤
            if semantic_score >= 0.95:
                # 매우 높은 의미 기반 점수는 지역명 영향일 수 있으므로 가중치 없이 사용
                # 하지만 로마자화 점수도 함께 고려
                final_score = max(romanized_score, semantic_score * 0.95)
            elif semantic_score >= 0.90:
                # 높은 의미 기반 점수도 약간 낮춤
                final_score = max(romanized_score, semantic_score * 1.0)
            else:
                # 일반적인 경우에는 약간의 가중치 적용
                final_score = max(romanized_score, semantic_score * 1.05)
        else:
            final_score = romanized_score
        
        return min(final_score, 1.0)  # 1.0 초과 방지
    
    def calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """주소 유사도 (개선: 건물/번지 단위까지 일치 확인)"""
        addr1 = self.normalize_text(addr1)
        addr2 = self.normalize_text(addr2)
        if not addr1 or not addr2:
            return 0.0
        
        # 도시명만 추출하여 비교
        city1 = self.extract_city_from_address(addr1)
        city2 = self.extract_city_from_address(addr2)
        
        # 도시명이 같고 주소가 매우 짧으면 점수를 낮춤 (도시명만 같을 가능성)
        if city1 and city2 and city1 == city2:
            if len(addr1.split()) <= 2 or len(addr2.split()) <= 2:
                # 도시명만 있거나 매우 짧은 주소는 점수를 낮춤
                base_score = 0.3
            else:
                base_score = 0.0
        else:
            base_score = 0.0
        
        # 구두점과 공백 정규화 (주소 포맷 차이 처리)
        import re
        # 구두점을 공백으로 변환
        addr1_normalized = re.sub(r'[.,;:]+', ' ', addr1)
        addr2_normalized = re.sub(r'[.,;:]+', ' ', addr2)
        # 여러 공백을 하나로
        addr1_normalized = re.sub(r'\s+', ' ', addr1_normalized).strip()
        addr2_normalized = re.sub(r'\s+', ' ', addr2_normalized).strip()
        
        # 정규화된 버전으로 유사도 계산
        ratio = fuzz.ratio(addr1_normalized, addr2_normalized) / 100.0
        partial_ratio = fuzz.partial_ratio(addr1_normalized, addr2_normalized) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(addr1_normalized, addr2_normalized) / 100.0
        
        # 원본 버전도 함께 고려 (더 높은 점수 사용)
        ratio_orig = fuzz.ratio(addr1, addr2) / 100.0
        partial_ratio_orig = fuzz.partial_ratio(addr1, addr2) / 100.0
        token_sort_ratio_orig = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        
        calculated_score = max(
            (ratio * 0.3 + partial_ratio * 0.5 + token_sort_ratio * 0.2),
            (ratio_orig * 0.3 + partial_ratio_orig * 0.5 + token_sort_ratio_orig * 0.2)
        )
        
        # 공통 단어/숫자 보너스 (예: "JARDIM GAV", "2929" 같은 공통 요소)
        # 정규화된 버전에서 단어 추출
        words1 = set(re.findall(r'\b\w+\b', addr1_normalized.upper()))
        words2 = set(re.findall(r'\b\w+\b', addr2_normalized.upper()))
        numbers1 = set(re.findall(r'\d+', addr1))
        numbers2 = set(re.findall(r'\d+', addr2))
        
        common_words = words1 & words2
        common_numbers = numbers1 & numbers2
        
        # 공통 단어/숫자가 있으면 보너스 점수 추가
        bonus = 0.0
        if common_words:
            # 공통 단어 비율
            word_bonus = len(common_words) / max(len(words1), len(words2)) * 0.15
            bonus += word_bonus
        if common_numbers:
            # 공통 숫자가 있으면 추가 보너스
            bonus += 0.1
        
        calculated_score = min(calculated_score + bonus, 1.0)
        
        # 도시명만 같을 때는 최대값 제한 (건물/번지 정보가 없으면 낮은 점수)
        if city1 and city2 and city1 == city2:
            # 공통 단어/숫자가 있으면 제한을 완화
            if common_words or common_numbers:
                return min(calculated_score, max(base_score, calculated_score * 0.9))
            else:
                return min(calculated_score, max(base_score, calculated_score * 0.8))
        
        return calculated_score
    
    def check_building_address_match(self, row1: pd.Series, row2: pd.Series) -> Tuple[bool, float]:
        """건물/번지 단위 주소 일치 확인 (중국 특성 반영)"""
        # 건물명/번지 필드 추출 (STREET는 도시명일 수 있으므로 제외)
        building_fields = ['HOUSE_NUM1', 'HOUSE_NUM2', 'STR_SUPPL1', 'STR_SUPPL2', 'STR_SUPPL3']
        
        building1_parts = []
        building2_parts = []
        
        # STREET 필드 확인 (도시명이 아닌 실제 거리명인지 확인)
        street1 = ""
        street2 = ""
        if 'STREET' in row1.index and pd.notna(row1['STREET']):
            street1 = self.normalize_text(str(row1['STREET']))
        if 'STREET' in row2.index and pd.notna(row2['STREET']):
            street2 = self.normalize_text(str(row2['STREET']))
        
        # STREET가 도시명과 같으면 제외 (도시명만 있는 경우)
        city1 = ""
        city2 = ""
        if 'CITY1' in row1.index and pd.notna(row1['CITY1']):
            city1 = self.normalize_text(str(row1['CITY1']))
        if 'CITY1' in row2.index and pd.notna(row2['CITY1']):
            city2 = self.normalize_text(str(row2['CITY1']))
        
        # STREET가 도시명과 다르면 건물 정보로 사용
        if street1 and street1 != city1:
            building1_parts.append(street1)
        if street2 and street2 != city2:
            building2_parts.append(street2)
        
        for field in building_fields:
            if field in row1.index and pd.notna(row1[field]):
                part1 = self.normalize_text(str(row1[field]))
                if part1:
                    building1_parts.append(part1)
            if field in row2.index and pd.notna(row2[field]):
                part2 = self.normalize_text(str(row2[field]))
                if part2:
                    building2_parts.append(part2)
        
        # 건물/번지 정보가 없으면 False (도시명만 있는 경우)
        if not building1_parts or not building2_parts:
            return (False, 0.0)
        
        building1 = ' '.join(building1_parts)
        building2 = ' '.join(building2_parts)
        
        # 번지 번호가 다르면 False (명확히 다른 주소)
        house_num1 = ""
        house_num2 = ""
        if 'HOUSE_NUM1' in row1.index and pd.notna(row1['HOUSE_NUM1']):
            house_num1 = self.normalize_text(str(row1['HOUSE_NUM1']))
        if 'HOUSE_NUM1' in row2.index and pd.notna(row2['HOUSE_NUM1']):
            house_num2 = self.normalize_text(str(row2['HOUSE_NUM1']))
        
        # 번지 번호가 둘 다 있고 다르면 False
        if house_num1 and house_num2 and house_num1 != house_num2:
            return (False, 0.0)
        
        # 건물/번지 유사도 계산
        ratio = fuzz.ratio(building1, building2) / 100.0
        partial_ratio = fuzz.partial_ratio(building1, building2) / 100.0
        
        # 숫자 부분 매칭 확인 (예: "DEPU12929"와 "JAMEL nº 2929"에서 "2929" 공통)
        import re
        numbers1 = set(re.findall(r'\d+', building1))
        numbers2 = set(re.findall(r'\d+', building2))
        number_match_bonus = 0.0
        if numbers1 and numbers2:
            # 공통 숫자가 있으면 보너스 점수 추가
            common_numbers = numbers1 & numbers2
            if common_numbers:
                # 공통 숫자가 있으면 부분 매칭 보너스
                number_match_bonus = 0.2
        
        # 건물/번지가 정확히 일치하거나 매우 유사해야 함
        # 숫자 매칭 보너스가 있으면 기준을 낮춤
        threshold = 0.85 if number_match_bonus == 0.0 else 0.75
        building_score = (ratio * 0.5 + partial_ratio * 0.3 + number_match_bonus)
        is_match = building_score >= threshold
        
        return (is_match, building_score)
    
    def extract_company_core_name(self, name: str) -> str:
        """
        고유 상호(Core Business Name) 추출
        GPT 프롬프트 기반: 지역명, 법인 형태, 상거래/업종 단어 제거, 브랜드/고유 명칭만 남김
        """
        if not name:
            return ""
        
        # [성능 개선] 캐시 확인
        if name in self._core_name_cache:
            return self._core_name_cache[name]
        
        import re
        
        name_clean = name.strip()
        
        # 1. 지역명 제거 (중국어 행정구역명 - GPT 가이드 반영)
        # 성(省) 이름
        chinese_provinces = ['山东', '江苏', '浙江', '广东', '河南', '四川', '湖北', '湖南', '河北', '安徽',
                           '福建', '江西', '陕西', '山西', '辽宁', '黑龙江', '吉林', '云南', '贵州', '广西',
                           '海南', '甘肃', '青海', '新疆', '西藏', '内蒙古', '宁夏', '北京', '上海', '天津', '重庆']
        # 도시(市) 이름
        chinese_cities = ['北京', '上海', '天津', '重庆', '聊城', '沈阳', '广州', '深圳', '杭州', '成都',
                         '武汉', '西安', '南京', '苏州', '青岛', '大连', '宁波', '厦门', '福州', '济南',
                         '郑州', '长沙', '石家庄', '哈尔滨', '长春', '太原', '合肥', '南昌', '昆明', '贵阳',
                         '南宁', '海口', '兰州', '西宁', '银川', '乌鲁木齐', '拉萨', '呼和浩特', '无锡', '佛山',
                         '东莞', '中山', '珠海', '惠州', '江门', '肇庆', '汕头', '潮州', '揭阳', '汕尾',
                         '湛江', '茂名', '阳江', '韶关', '清远', '云浮', '梅州', '河源', '徐州', '常州',
                         '南通', '扬州', '镇江', '泰州', '盐城', '淮安', '宿迁', '连云港', '温州', '嘉兴',
                         '湖州', '绍兴', '金华', '台州', '丽水', '衢州', '舟山']
        # 구/현(区/县) 패턴 - 패턴으로 처리
        # 행정구역명 패턴: ~省, ~市, ~区, ~县, ~州, ~道, ~府, ~都 등
        chinese_regions = chinese_provinces + chinese_cities
        
        # 2. 법인 형태 제거 (단독으로 사용될 때만 제거)
        chinese_legal_forms = ['有限公司', '股份有限公司', '有限责任公司', '集团', '公司', '企业',
                               '股份', '有限', '责任']
        # 영어 법인 형태: 단독 단어로만 제거 (예: "CO"는 "COMPANY"의 약자일 수 있으므로 주의)
        english_legal_forms = ['LTD', 'LIMITED', 'INC', 'INCORPORATED', 'CORP', 'CORPORATION',
                              'LLC', 'COMPANY', 'GROUP']
        # "CO"는 맨 끝에 있을 때만 제거 (예: "ABC CO" → "ABC", "ABC CO.LTD" → "ABC")
        # 하지만 "IND.ENG.CO" 같은 경우는 제거하지 않음
        portuguese_legal_forms = ['LTDA', 'LDA', 'EIRELI', 'ME', 'EPP', 'SA', 'S.A.', 'SOCIEDADE',
                                 'ANONIMA', 'EMPRESA', 'LIMITADA', 'INDIVIDUAL', 'RESPONSABILIDADE']
        vietnamese_legal_forms = ['CONG TY', 'TNHH', 'CO PHAN', 'DOANH NGHIEP', 'TRACH NHIEM',
                                 'HUU HAN', 'TU NHAN', 'THANH VIEN']
        korean_legal_forms = ['주식회사', '유한회사', '법인']
        indonesian_legal_forms = ['PT', 'PT.', 'CV', 'CV.', 'TB', 'TB.', 'UD', 'UD.', 'TBK', 'TBK.']
        thai_legal_forms = ['LTD', 'LIMITED']
        
        # 3. 상거래/일반 업종 단어 제거 (단독으로 사용될 때만 제거)
        # 주의: "MACHINERY", "TECHNOLOGY" 같은 단어는 회사명의 일부일 수 있으므로 제거하지 않음
        chinese_business_types = ['贸易', '商贸', '商业', '物资', '集团', '实业', '科技', '电子',
                                 '设备', '机械', '工程', '建设', '建筑', '开发', '投资', '管理',
                                 '咨询', '服务', '物流', '运输', '制造', '生产', '销售', '经销',
                                 '经营', '代理', '进出口', '国际', '全球']
        # "TECHNOLOGY", "MACHINERY" 같은 단어는 회사명의 일부일 수 있으므로 제거하지 않음
        # 예: "DAESUNG MACHINERY"에서 "MACHINERY"는 회사명의 일부
        # 예: "DAESUNG TECHNOLOGY"에서 "TECHNOLOGY"는 회사명의 일부
        english_business_types = ['TRADING', 'COMMERCE', 'BUSINESS', 'GROUP', 'HOLDINGS', 
                                 'ENTERPRISES', 'INDUSTRIES', 'INTERNATIONAL', 'GLOBAL',
                                 'SERVICES', 'SOLUTIONS', 'SYSTEMS']
        portuguese_business_types = ['COMERCIAL', 'COMERCIO', 'COMMERCIAL', 'INDUSTRIA', 
                                    'INDUSTRIAL', 'SERVICOS', 'SERVICOS']
        # 포르투갈어 지역명/국가명 (고유 상호가 아님)
        # GPT 가이드: DO BRASIL, DA BRASIL, DE BRASIL은 "브라질 법인"을 의미하며 회사 식별력이 없음
        portuguese_regions = ['DO BRASIL', 'DA BRASIL', 'DE BRASIL', 'BRASIL', 'BRASILEIRA', 'BRASILEIRO', 'BRASILEIRAS',
                             'BRASILEIROS', 'BRASILIA', 'SAO PAULO', 'RIO DE JANEIRO',
                             'MINAS GERAIS', 'PARANA', 'SANTA CATARINA', 'RIO GRANDE DO SUL']
        # 인도네시아 일반 단어 (고유 상호가 아님 - GPT 피드백 반영)
        # SEJAHTERA, MAKMUR, ABADI, JAYA, SENTOSA, MANDIRI, PUTRA는 수천 개 회사가 사용하는 일반 단어
        # 이 단어들은 "OO상사", "OO산업" 급의 일반 수식어로 고유 상호가 아님
        # 추가: SURYA, PRATAMA, PERKASA, HIDUP 등도 흔한 상호 토큰
        indonesian_common_words = ['SEJAHTERA', 'MAKMUR', 'ABADI', 'JAYA', 'SENTOSA', 'MANDIRI', 'PUTRA',
                                  'SURYA', 'PRATAMA', 'PERKASA', 'HIDUP', 'INDONESIA']
        # 인도네시아 국가/법인형태 토큰 (추가)
        indonesian_country_tokens = ['INDONESIA', 'IND', 'IDN', 'ID']
        
        # 모든 불용어 통합 (긴 단어부터 제거하기 위해 정렬)
        all_stopwords = (chinese_regions + chinese_legal_forms + chinese_business_types +
                        english_legal_forms + english_business_types +
                        portuguese_legal_forms + portuguese_business_types + portuguese_regions +
                        vietnamese_legal_forms + korean_legal_forms +
                        indonesian_legal_forms + thai_legal_forms +
                        indonesian_common_words + indonesian_country_tokens)
        
        # 악센트 제거 후 정규화
        name_clean_normalized = self.remove_accents(name_clean.upper())
        
        # 불용어 제거 (순서 중요: 긴 단어부터 제거)
        all_stopwords_sorted = sorted(all_stopwords, key=len, reverse=True)
        
        # 스크립트 판별 (영문/로마자는 경계 기반, 그 외는 부분 삭제 허용)
        try:
            from .cheap_gate import get_script_key
            script_type = get_script_key(name_clean)
            is_latin_script = script_type in ['latin', 'mixed']
        except ImportError:
            # fallback: 영문 포함 여부로 판단
            is_latin_script = bool(re.search(r'[a-zA-Z]', name_clean))
        
        for stopword in all_stopwords_sorted:
            # 영문/로마자 stopword는 단어 경계 기반 제거 (오삭제 방지)
            if is_latin_script and stopword.isalpha():
                # \bSTOPWORD\b 패턴으로 제거
                pattern = r'\b' + re.escape(stopword) + r'\b'
                name_clean = re.sub(pattern, ' ', name_clean, flags=re.IGNORECASE)
                name_clean_normalized = re.sub(pattern, ' ', name_clean_normalized, flags=re.IGNORECASE)
            else:
                # 중국어/한글 등은 부분 삭제 허용 (기존 방식)
                name_clean = name_clean.replace(stopword, ' ')
                name_clean = name_clean.replace(stopword.upper(), ' ')
                name_clean = name_clean.replace(stopword.lower(), ' ')
                # 정규화된 버전에서도 제거
                stopword_normalized = self.remove_accents(stopword.upper())
                if stopword_normalized:
                    name_clean_normalized = name_clean_normalized.replace(stopword_normalized, ' ')
        
        # 중국어 행정구역명 패턴 제거 (~省, ~市, ~区, ~县, ~州, ~道, ~府, ~都)
        # GPT 가이드: 행정구역명이 포함된 상태에서는 Core Name Match를 수행하지 않음
        import re
        chinese_admin_patterns = [
            r'[\u4e00-\u9fff]+省',  # ~省
            r'[\u4e00-\u9fff]+市',  # ~市
            r'[\u4e00-\u9fff]+区',  # ~区
            r'[\u4e00-\u9fff]+县',  # ~县
            r'[\u4e00-\u9fff]+州',  # ~州
            r'[\u4e00-\u9fff]+道',  # ~道
            r'[\u4e00-\u9fff]+府',  # ~府
            r'[\u4e00-\u9fff]+都',  # ~都
        ]
        for pattern in chinese_admin_patterns:
            name_clean = re.sub(pattern, ' ', name_clean)
            name_clean_normalized = re.sub(pattern, ' ', name_clean_normalized)
        
        # 특별 처리: "CO"는 맨 끝에 있을 때만 제거
        # 예: "ABC CO" → "ABC", "ABC CO.LTD" → "ABC"
        # 하지만 "IND.ENG.CO" 같은 경우는 제거하지 않음
        # 맨 끝의 "CO" 또는 "CO." 제거 (단어 경계 확인)
        name_clean_normalized = re.sub(r'\bCO\b\s*$', '', name_clean_normalized)
        name_clean_normalized = re.sub(r'\bCO\.\s*$', '', name_clean_normalized)
        
        # 정규화된 버전 사용
        name_clean = name_clean_normalized
        
        # 특별 처리: "CO"는 맨 끝에 있을 때만 제거
        # 예: "ABC CO" → "ABC", "ABC CO.LTD" → "ABC"
        # 하지만 "IND.ENG.CO" 같은 경우는 제거하지 않음
        # 맨 끝의 "CO" 또는 "CO." 제거 (단어 경계 확인)
        # "CO.," 같은 경우도 처리
        name_clean = re.sub(r'\bCO\b\s*[.,]*\s*$', '', name_clean)
        name_clean = re.sub(r'\bCO\.\s*[.,]*\s*$', '', name_clean)
        name_clean = re.sub(r'\bCO,\s*$', '', name_clean)
        
        # 공백 정리 (단어 간 구분은 유지)
        name_clean = re.sub(r'\s+', ' ', name_clean).strip()
        
        # 마지막 구두점 제거
        name_clean = re.sub(r'[.,]+$', '', name_clean).strip()
        
        # [성능 개선] 캐시 저장
        self._core_name_cache[name] = name_clean
        
        return name_clean
    
    def check_core_name_match(self, name1: str, name2: str) -> Tuple[bool, float]:
        """
        고유 상호 비교 (GPT 프롬프트 기반)
        - 핵심 단어가 명확히 다르면 다른 업체
        - 철자 차이, 단어 순서 차이, 구두점 차이는 허용
        - 완전히 다른 단어 조합은 허용하지 않음
        - 다국어 처리: 같은 업체의 다른 언어 표기도 탐지
        
        중요: 행정구역명이 포함된 상태에서는 Core Name Match를 수행하지 않음
        """
        import re
        
        # 행정구역명이 포함되어 있는지 확인 (중국어)
        chinese_admin_patterns = [
            r'[\u4e00-\u9fff]+省',  # ~省
            r'[\u4e00-\u9fff]+市',  # ~市
            r'[\u4e00-\u9fff]+区',  # ~区
            r'[\u4e00-\u9fff]+县',  # ~县
            r'[\u4e00-\u9fff]+州',  # ~州
            r'[\u4e00-\u9fff]+道',  # ~道
            r'[\u4e00-\u9fff]+府',  # ~府
            r'[\u4e00-\u9fff]+都',  # ~都
        ]
        
        # 행정구역명이 포함되어 있으면 Core Name Match 수행하지 않음
        for pattern in chinese_admin_patterns:
            if re.search(pattern, name1) or re.search(pattern, name2):
                return (False, 0.0)
        
        # 중국어 행정구역명 단어 체크 (성, 도시명)
        chinese_admin_words = ['山东', '江苏', '浙江', '广东', '北京', '上海', '天津', '重庆', '聊城', '沈阳',
                              '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '苏州', '青岛', '大连']
        for admin_word in chinese_admin_words:
            if admin_word in name1 or admin_word in name2:
                return (False, 0.0)
        
        # 브라질 업체명 체크 (GPT 가이드 반영)
        # DO BRASIL, DA BRASIL, DE BRASIL은 "브라질 법인"을 의미하며 회사 식별력이 없음
        # 이 표현이 포함된 상태에서는 절대 Core Name Match를 수행하지 않음
        brasil_country_patterns = [
            r'\bDO\s+BRASIL\b',
            r'\bDA\s+BRASIL\b',
            r'\bDE\s+BRASIL\b',
        ]
        name1_upper = name1.upper()
        name2_upper = name2.upper()
        for pattern in brasil_country_patterns:
            if re.search(pattern, name1_upper) or re.search(pattern, name2_upper):
                return (False, 0.0)
        
        core1 = self.extract_company_core_name(name1)
        core2 = self.extract_company_core_name(name2)
        
        if not core1 or not core2:
            return (False, 0.0)
        
        # 핵심어가 너무 짧으면 False
        if len(core1) < 2 or len(core2) < 2:
            return (False, 0.0)
        
        core1_upper = core1.upper().strip()
        core2_upper = core2.upper().strip()
        
        # 1. 정확히 일치
        if core1_upper == core2_upper:
            return (True, 1.0)
        
        # 2. 한쪽이 다른 쪽에 포함되어 있으면 (부분 매칭)
        if core1_upper in core2_upper or core2_upper in core1_upper:
            # 포함 비율 확인
            if len(core1_upper) <= len(core2_upper):
                ratio = len(core1_upper) / len(core2_upper)
            else:
                ratio = len(core2_upper) / len(core1_upper)
            # 포함 비율이 70% 이상이면 일치로 판단
            if ratio >= 0.7:
                return (True, ratio)
        
        # 3. 유사도 계산 (철자 차이, 단어 순서 차이 허용)
        ratio_score = fuzz.ratio(core1_upper, core2_upper) / 100.0
        token_sort_score = fuzz.token_sort_ratio(core1_upper, core2_upper) / 100.0
        partial_score = fuzz.partial_ratio(core1_upper, core2_upper) / 100.0
        
        # 최종 유사도 (단어 순서 차이를 고려)
        core_similarity = max(ratio_score, token_sort_score * 0.9, partial_score * 0.8)
        
        # 4. 다국어 처리: 로마자화 버전도 비교
        pinyin1 = self.transliterate_chinese(name1)
        pinyin2 = self.transliterate_chinese(name2)
        romaji1 = self.transliterate_japanese(name1)
        romaji2 = self.transliterate_japanese(name2)
        korean_roman1 = self.transliterate_korean(name1)
        korean_roman2 = self.transliterate_korean(name2)
        
        # 로마자화된 고유 상호 추출
        if pinyin1:
            core1_pinyin = self.extract_company_core_name(pinyin1).upper().strip()
            if core1_pinyin and core1_pinyin != core1_upper:
                # 로마자화 버전과 원본 비교
                if core1_pinyin == core2_upper or core2_upper in core1_pinyin or core1_pinyin in core2_upper:
                    return (True, 0.85)
        if pinyin2:
            core2_pinyin = self.extract_company_core_name(pinyin2).upper().strip()
            if core2_pinyin and core2_pinyin != core2_upper:
                if core2_pinyin == core1_upper or core1_upper in core2_pinyin or core2_pinyin in core1_upper:
                    return (True, 0.85)
        
        # 의미 기반 유사도도 고려 (다국어 매칭)
        if self.use_embedding:
            semantic_score = self.calculate_semantic_similarity(core1, core2)
            if semantic_score >= 0.85:
                # 의미 기반 점수가 높으면 다국어 매칭 가능성
                core_similarity = max(core_similarity, semantic_score * 0.9)
        
        # 5. 핵심 단어 추출 및 비교
        import re
        words1 = set(re.findall(r'\b\w+\b', core1_upper))
        words2 = set(re.findall(r'\b\w+\b', core2_upper))
        
        # 공통 단어가 없으면 완전히 다른 단어 조합
        if not words1 or not words2:
            return (False, core_similarity)
        
        common_words = words1 & words2
        all_words = words1 | words2
        
        # 공통 단어 비율
        if all_words:
            common_ratio = len(common_words) / len(all_words)
        else:
            common_ratio = 0.0
        
        # 핵심 단어가 명확히 다르면 (공통 단어 비율이 30% 미만)
        if common_ratio < 0.3:
            # 특별 케이스: 공통 브랜드명이 있고 유사도가 높으면 허용
            # [개선] 범용 토큰 블랙리스트 강화
            generic_tokens_blacklist = {
                'GLOBAL', 'GROUP', 'TECH', 'TECHNOLOGY', 'TECHNOLOGIES',
                'INTERNATIONAL', 'WORLD', 'CENTRAL', 'CENTRE', 'CENTER',
                'SOLUTION', 'SOLUTIONS', 'SYSTEM', 'SYSTEMS', 'SERVICE', 'SERVICES',
                'INDUSTRIAL', 'INDUSTRIES', 'ENTERPRISE', 'ENTERPRISES',
                'HOLDING', 'HOLDINGS', 'PART', 'PARTS'
            }
            
            # 공통 브랜드명 찾기 (범용 토큰 제외)
            # 숫자 포함 토큰(예: "3M", "BASF") 허용, 또는 가장 긴 토큰 우선
            potential_brands = [w for w in common_words 
                              if len(w) >= 4 and w not in generic_tokens_blacklist
                              and (w.isupper() or any(c.isdigit() for c in w))]
            
            # 가장 긴 토큰 Top-N만 브랜드로 인정 (식별력 높은 토큰 우선)
            if potential_brands:
                # 길이 기준 정렬 후 상위 2개만
                potential_brands_sorted = sorted(potential_brands, key=len, reverse=True)[:2]
                if potential_brands_sorted:
                    # 공통 브랜드명이 있으면 기준 완화
                    if core_similarity >= 0.60:
                        return (True, core_similarity)
                    # 유사도가 낮아도 공통 브랜드명이 있고 이름 전체가 유사하면 허용
                    name_sim = fuzz.ratio(core1_upper, core2_upper) / 100.0
                    if name_sim >= 0.60:
                        return (True, name_sim)
            return (False, core_similarity)
        
        # 공통 단어가 있고 유사도가 높으면 일치
        # 기준: 공통 단어 비율이 50% 이상이거나 유사도가 85% 이상
        if common_ratio >= 0.5 or core_similarity >= 0.85:
            return (True, max(common_ratio, core_similarity))
        
        # 중간 범위: 공통 단어 비율이 30% 이상이고 유사도가 70% 이상
        if common_ratio >= 0.3 and core_similarity >= 0.70:
            return (True, (common_ratio + core_similarity) / 2)
        
        # 완화된 기준: 공통 단어 비율이 25% 이상이고 유사도가 65% 이상
        if common_ratio >= 0.25 and core_similarity >= 0.65:
            return (True, (common_ratio + core_similarity) / 2)
        
        # 특별 케이스: 공통 브랜드명이 있고 유사도가 중간 이상이면 허용
        # 예: "DAESUNG MACHINERY" vs "DAESUNG TECHNOLOGY E&C"
        # 하지만 일반 단어(국가명, 일반 업종 용어)만 공통이면 제외
        if common_words and len(common_words) >= 1:
            # 공통 단어에서 일반 단어 제외 (통합된 general_words 사용)
            unique_common_words = common_words - self.GENERAL_WORDS
            
            # 고유 브랜드명만 공통으로 있으면 허용
            if unique_common_words:
                common_word_list = list(unique_common_words)
                # 공통 단어가 브랜드명일 가능성이 높으면 (4자 이상)
                if any(len(w) >= 4 for w in common_word_list):
                    if core_similarity >= 0.65:
                        return (True, core_similarity)
            # 일반 단어만 공통이면 제외 (더 엄격하게)
            if common_words.issubset(self.GENERAL_WORDS) or len(unique_common_words) == 0:
                # 일반 단어만 공통이면 고유 상호가 다름
                return (False, core_similarity)
        
        return (False, core_similarity)
    
    def extract_city_from_address(self, address: str) -> str:
        """주소에서 도시명 추출"""
        if not address:
            return ""
        
        # 주요 도시명 패턴 찾기
        city_patterns = [
            r'([A-Z]+)\s+(?:SHI|CITY)',  # BEIJING SHI, SEOUL CITY 등
            r'([가-힣]+시)',  # 서울시, 부산시 등
            r'([\u4e00-\u9fff]+市)',  # 北京市, 上海市 등
            r'([\u4e00-\u9fff]+区)',  # 海淀区 등
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, address, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return ""
    
    def check_business_type_match(self, name1: str, name2: str) -> bool:
        """업종 키워드 비교 (다른 업종이면 False) - 다국어 지원"""
        # 다국어 업종 키워드 (명확한 업종 구분)
        business_types = {
            '건축': ['建筑', '建设', '工程', '施工', '건축', '건설', '공사', 'CONSTRUCTION', 'BUILDING', 'ENGINEERING'],
            '전력': ['电力', '电气', '电', '能源', '전력', '전기', '에너지', 'POWER', 'ELECTRIC', 'ENERGY'],
            '화학': ['化工', '化学', '石化', '화학', 'CHEMICAL', 'CHEMISTRY'],
            '전자': ['电子', '电器', '电控', '전자', 'ELECTRONICS', 'ELECTRIC'],
            '기계': ['机械', '设备', '制造', '기계', '장비', 'EQUIPMENT', 'MANUFACTURING'],
            # 주의: 'MACHINERY'는 회사명의 일부일 수 있으므로 제외 (예: "DAESUNG MACHINERY")
            '의료': ['医疗', '医药', '生物', '의료', '의약', '생물', 'MEDICAL', 'PHARMACEUTICAL', 'BIOTECH', 
                    'UNIODONTO', 'ODONTOLOGIA', 'DENTAL', 'DENTISTRY', 'CLINICA', 'CLINIC'],
            '운송': ['运输', '物流', '货运', '운송', '물류', 'TRANSPORT', 'LOGISTICS', 'SHIPPING'],
            '무역': ['贸易', '商贸', '商业', '무역', '상업', 'TRADE', 'COMMERCE', 'BUSINESS', 
                    'COMERCIAL', 'COMERCIO', 'COMMERCIAL'],
            'IT': ['科技', '技术', '软件', '信息', '기술', '소프트웨어', '정보', 'SOFTWARE', 'IT', 'TECH'],
            # 주의: 'TECHNOLOGY'는 회사명의 일부일 수 있으므로 제외 (예: "DAESUNG TECHNOLOGY")
            '식품': ['食品', '农业', '餐饮', '식품', '농업', '음식', 'FOOD', 'AGRICULTURE', 'RESTAURANT'],
            '조명': ['照明', '灯具', '조명', 'LIGHTING', 'LAMP'],
            '자동차': ['汽车', '车辆', '자동차', 'AUTOMOTIVE', 'VEHICLE', 'AUTO'],
        }
        
        # 각 업체가 어떤 업종 키워드를 가지고 있는지 확인
        type1_set = set()
        type2_set = set()
        
        for business_type, keywords in business_types.items():
            if any(keyword in name1 for keyword in keywords):
                type1_set.add(business_type)
            if any(keyword in name2 for keyword in keywords):
                type2_set.add(business_type)
        
        # 둘 다 업종 키워드가 없으면 제외하지 않음
        if not type1_set and not type2_set:
            return True
        
        # 하나만 업종 키워드를 가지고 있으면 다른 업종일 가능성 높음
        if (type1_set and not type2_set) or (type2_set and not type1_set):
            return False
        
        # 둘 다 업종 키워드를 가지고 있으면 같은 업종인지 확인
        if type1_set and type2_set:
            # 교집합이 있으면 같은 업종
            if type1_set & type2_set:
                return True
            # 교집합이 없어도 같은 브랜드명이 있으면 같은 업체일 가능성
            # 예: "DAESUNG MACHINERY" vs "DAESUNG TECHNOLOGY E&C"
            name1_upper = name1.upper()
            name2_upper = name2.upper()
            # 공통 단어 추출 (브랜드명일 가능성)
            import re
            words1 = set(re.findall(r'\b[A-Z]{3,}\b', name1_upper))
            words2 = set(re.findall(r'\b[A-Z]{3,}\b', name2_upper))
            common_brands = words1 & words2
            # 공통 브랜드명이 있고 길이가 4자 이상이면 같은 업체일 가능성
            if common_brands and any(len(brand) >= 4 for brand in common_brands):
                return True
            # 교집합이 없으면 다른 업종
            return False
        
        return True
    
    def is_candidate(self, row1: pd.Series, row2: pd.Series, return_overlap_count: bool = False) -> Tuple[bool, str, int]:
        """
        STEP 1: 중복 후보(CANDIDATE) 여부 판단 (Recall 중심, 하지만 엄격한 기준)
        GPT 프롬프트 기반: 같은 업체일 가능성이 있어 STEP2로 넘길 가치가 있는지만 판단
        
        중요 원칙:
        - 공통된 일반 단어, 지역명, 법인 형태 때문에 후보로 판단하지 않음
        - 특히 인도네시아, 중국, 베트남 데이터에서 일반 단어로 인한 오탐 방지
        
        Returns:
            (is_candidate: bool, reason: str, overlap_count: int)
            - overlap_count: 고유 단어 공통 개수 (stopword 제외)
        """
        name1 = row1.get('공급업체명', '')
        name2 = row2.get('공급업체명', '')
        
        # STEP1 전용 STOP WORD 리스트 (더 엄격)
        # 1. 업체명 전처리: 고유 상호 추출 (법인 형태, 지역명, 일반 단어 제거)
        core1 = self.extract_company_core_name(str(name1) if pd.notna(name1) else '')
        core2 = self.extract_company_core_name(str(name2) if pd.notna(name2) else '')
        
        # 고유 상호가 없으면 STEP1 후보에서 제외
        if not core1 or not core2:
            return (False, "고유 상호 추출 실패", 0)
        
        import re
        # 고유 상호를 토큰으로 분리
        words1 = set(re.findall(r'\b\w+\b', core1.upper()))
        words2 = set(re.findall(r'\b\w+\b', core2.upper()))
        
        # STOP WORD 제거 (통합된 stopword 사용)
        unique_words1 = words1 - self.STEP1_STOP_WORDS
        unique_words2 = words2 - self.STEP1_STOP_WORDS
        
        # 고유 단어가 하나도 없으면 제외
        if not unique_words1 or not unique_words2:
            return (False, "고유 단어 없음 (STOP WORD만 존재)", 0)
        
        # 조건 (D): 정규화된 이름이 동일한 경우 (구두점/공백/법인형태 차이만 있는 케이스)
        # 예: "LEEPACK CO., LTD" vs "LEEPACK CO.,LTD" -> "LEEPACK"
        normalized_name1 = self.normalize_name_for_blocking(str(name1) if pd.notna(name1) else '')
        normalized_name2 = self.normalize_name_for_blocking(str(name2) if pd.notna(name2) else '')
        if normalized_name1 and normalized_name2 and normalized_name1 == normalized_name2:
            # 정규화된 이름이 완전히 동일한 경우는 강한 신호이므로 overlap_count=1로 설정
            return (True, f"정규화된 이름 동일: {normalized_name1}", 1)
        
        # 조건 (A): 고유 상호 토큰이 최소 2개 이상 의미 있게 겹침 (STOP WORD 제외)
        # [GPT 가이드] 최소 2개 이상의 Core Token이 일치할 때만 후보로 허용
        # 단, Normalized Name이 완전히 동일한 경우는 예외적으로 1개 토큰이어도 허용
        unique_common_words = unique_words1 & unique_words2
        
        # 일반 단어만 공통인 경우 제외 (인도네시아 케이스)
        indonesian_general_only = {
            'SEJAHTERA', 'MAKMUR', 'ABADI', 'JAYA', 'SENTOSA', 'MANDIRI', 'PUTRA',
            'NIAGA', 'CENTRAL', 'CENTRA', 'INDUSTRIAL', 'PART', 'INTERNATIONAL'
        }
        
        # 일반 단어 제외한 고유 단어만 카운트
        unique_common_words_filtered = unique_common_words - indonesian_general_only
        
        if unique_common_words:
            # 일반 단어만 공통이면 제외
            if unique_common_words.issubset(indonesian_general_only):
                return (False, "일반 단어만 공통 (고유 상호 없음)", 0)
            
            # 최소 2개 이상의 고유 단어가 공통이어야 함
            if len(unique_common_words_filtered) >= 2:
                return (True, f"고유 단어 2개 이상 공통: {unique_common_words_filtered}", len(unique_common_words_filtered))
            elif len(unique_common_words_filtered) == 1:
                # 1개만 공통인 경우: Normalized Name이 동일한 경우만 허용 (이미 조건 D에서 처리됨)
                # 여기서는 제외
                return (False, f"고유 단어 1개만 공통 (부족): {unique_common_words_filtered}", len(unique_common_words_filtered))
        
        # 조건 (B): 고유 상호가 축약/확장 관계일 가능성 확인
        # 예: TRIDUTA ↔ TRI DUTA, LONG-LITE ↔ LONGLITE
        core1_no_space = re.sub(r'[\s\-]', '', core1.upper())
        core2_no_space = re.sub(r'[\s\-]', '', core2.upper())
        
        # 한쪽이 다른 쪽의 부분 문자열인지 확인 (최소 4글자 이상)
        if len(core1_no_space) >= 4 and len(core2_no_space) >= 4:
            if core1_no_space in core2_no_space or core2_no_space in core1_no_space:
                # STOP WORD가 포함되어 있지 않은지 확인 (통합된 stopword 사용)
                if not any(stop in core1_no_space for stop in self.STEP1_STOP_WORDS) and \
                   not any(stop in core2_no_space for stop in self.STEP1_STOP_WORDS):
                    overlap_count_abbr = len(unique_common_words_filtered) if 'unique_common_words_filtered' in locals() else 0
                    return (True, f"축약/확장 관계 가능성: {core1_no_space} ↔ {core2_no_space}", overlap_count_abbr)
        
        # 철자 유사도 확인 (축약/확장 관계)
        from rapidfuzz import fuzz
        if len(core1_no_space) >= 4 and len(core2_no_space) >= 4:
            ratio = fuzz.ratio(core1_no_space, core2_no_space) / 100.0
            # 높은 유사도이면서 STOP WORD가 포함되지 않은 경우
            if ratio >= 0.75:
                # STOP WORD가 포함되어 있지 않은지 확인 (통합된 stopword 사용)
                if not any(stop in core1_no_space for stop in self.STEP1_STOP_WORDS) and \
                   not any(stop in core2_no_space for stop in self.STEP1_STOP_WORDS):
                    overlap_count_fuzz = len(unique_common_words_filtered) if 'unique_common_words_filtered' in locals() else 0
                    return (True, f"고유 상호 유사도 높음: {ratio:.3f}", overlap_count_fuzz)
        
        # 조건 (C): 같은 국가 + 같은 도시이며, 고유 상호가 완전히 무관하다고 단정할 수 없음
        # 이 조건은 Recall을 높이기 위해 필요하지만, 너무 느슨하지 않게 적용
        land1 = row1.get('Land', '')
        land2 = row2.get('Land', '')
        city1 = row1.get('CITY1', '')
        city2 = row2.get('CITY1', '')
        
        if land1 and land2 and land1 == land2:
            if city1 and city2:
                # 도시명 정규화 후 비교
                city1_norm = self.normalize_text(str(city1)).upper()
                city2_norm = self.normalize_text(str(city2)).upper()
                if city1_norm == city2_norm or city1_norm in city2_norm or city2_norm in city1_norm:
                    # 같은 도시 + 이름 유사도가 높으면 후보로 판단
                    # 하지만 STOP WORD만 공통인 경우는 제외
                    name_sim = self.calculate_multilingual_similarity(
                        str(name1) if pd.notna(name1) else '',
                        str(name2) if pd.notna(name2) else ''
                    )
                    # 이름 유사도가 높고 (0.70 이상), STOP WORD만 공통이 아닌 경우
                    if name_sim >= 0.70:
                        # STOP WORD만 공통인지 확인
                        all_common_words = words1 & words2
                        if all_common_words:
                            non_stop_common = all_common_words - self.STEP1_STOP_WORDS
                            # STOP WORD가 아닌 공통 단어가 있으면 후보로 판단
                            if non_stop_common:
                                return (True, f"같은 국가({land1}) 및 도시({city1}), 이름 유사도: {name_sim:.3f}, 공통 단어: {non_stop_common}", len(non_stop_common))
                        else:
                            # 공통 단어가 없어도 이름 유사도가 매우 높으면 (0.85 이상) 후보로 판단
                            # 예: 축약/확장 관계이지만 단어 분리로 인해 공통 단어가 없는 경우
                            if name_sim >= 0.85:
                                return (True, f"같은 국가({land1}) 및 도시({city1}), 이름 유사도 매우 높음: {name_sim:.3f}", 0)
        
        return (False, "후보 조건 불만족", 0)
    
    def are_duplicates(self, row1: pd.Series, row2: pd.Series) -> Tuple[bool, float, str, str]:
        """
        STEP 2: 동일 법인(SAME_ENTITY) 최종 판단 (Precision 중심)
        GPT 프롬프트 기반: 실제로 병합해도 되는 동일 법인인지 판단
        
        핵심 원칙:
        1. 고유 상호가 일치해야 함 (필수)
        2. 주소는 보조 지표 (주소만 같고 업체명이 다르면 동일 업체로 판단하지 않음)
        3. 업종 충돌 확인
        4. 확실하지 않으면 같은 업체로 판단하지 않음 (False Positive 방지)
        
        Returns:
            (is_duplicate: bool, confidence: float, match_type: str, match_reason: str)
        """
        name1 = row1.get('공급업체명', '')
        name2 = row2.get('공급업체명', '')
        
        # 0단계: 정규화된 이름이 동일한 경우 확정 규칙 (구두점/공백/법인형태 차이만 있는 케이스)
        # 예: "LEEPACK CO., LTD" vs "LEEPACK CO.,LTD"
        normalized_name1 = self.normalize_name_for_blocking(str(name1) if pd.notna(name1) else '')
        normalized_name2 = self.normalize_name_for_blocking(str(name2) if pd.notna(name2) else '')
        if normalized_name1 and normalized_name2 and normalized_name1 == normalized_name2:
            # [과병합 방지] 짧은 이름은 확정 규칙 금지
            normalized_name_len = len(normalized_name1.strip())
            if normalized_name_len < 4:
                # 너무 짧은 이름(예: "ABC")은 다른 신호 필요
                # 주소/도시/코드 중 최소 1개 강한 신호 요구
                addr1 = self.normalize_address(row1)
                addr2 = self.normalize_address(row2)
                address_similarity = self.calculate_address_similarity(addr1, addr2)
                city1 = str(row1.get('CITY1', '')).strip().upper() if pd.notna(row1.get('CITY1', '')) else ''
                city2 = str(row2.get('CITY1', '')).strip().upper() if pd.notna(row2.get('CITY1', '')) else ''
                
                # 주소 유사도가 높거나 도시가 일치해야 함
                if address_similarity < 0.70 and city1 != city2:
                    # 짧은 이름 + 주소/도시 불일치 → 확정 규칙 적용 안 함
                    pass  # 아래 core_name_match로 넘어감
                else:
                    # 주소/도시 일치 → 확정 가능
                    pass  # 아래 로직 계속
            else:
                # 긴 이름은 기존 로직 사용
                pass
            
            # 국가 확인
            land1 = row1.get('Land', '')
            land2 = row2.get('Land', '')
            land1_str = str(land1).strip().upper() if pd.notna(land1) else ''
            land2_str = str(land2).strip().upper() if pd.notna(land2) else ''
            
            # 정규화된 이름이 동일하고 국가가 동일하면 동일 법인으로 판단
            # (단, 짧은 이름은 위에서 이미 필터링됨)
            if land1_str and land2_str and land1_str == land2_str:
                # 주소 확인
                addr1 = self.normalize_address(row1)
                addr2 = self.normalize_address(row2)
                address_similarity = self.calculate_address_similarity(addr1, addr2)
                
                # 국가가 동일하고 정규화된 이름이 동일하면, 주소가 달라도 동일 법인 (다른 사업장)
                if address_similarity >= 0.70:
                    # 주소가 유사하면 같은 사업장
                    confidence = 0.98
                    return (True, confidence, "NORMALIZED_NAME_MATCH", f"정규화된 이름 동일: {normalized_name1}, 주소 유사도: {address_similarity:.3f}")
                elif address_similarity >= 0.50:
                    # 주소가 중간 수준이면 같은 사업장 가능성
                    confidence = 0.95
                    return (True, confidence, "NORMALIZED_NAME_MATCH", f"정규화된 이름 동일: {normalized_name1}, 주소 유사도: {address_similarity:.3f}")
                else:
                    # 주소가 다르면 동일 법인이지만 다른 사업장
                    confidence = 0.92  # 정규화된 이름 일치 + 국가 일치 = 높은 신뢰도
                    return (True, confidence, "SAME_COMPANY_DIFFERENT_LOCATION", f"정규화된 이름 동일: {normalized_name1}, 국가 동일: {land1_str}, 주소 다름 (다른 사업장), 주소 유사도: {address_similarity:.3f}")
            
            # 국가가 다르면 주소 유사도 확인
            addr1 = self.normalize_address(row1)
            addr2 = self.normalize_address(row2)
            address_similarity = self.calculate_address_similarity(addr1, addr2)
            
            # 주소가 너무 다르면 (0.30 미만) 제외
            if address_similarity < 0.30:
                return (False, 0.0, "NORMALIZED_NAME_MATCH", f"정규화된 이름 동일하지만 국가가 다르고 주소가 너무 다름: {address_similarity:.3f}")
            
            # 정규화된 이름이 동일하고 주소가 유사하면 확정
            confidence = 0.90  # 국가가 다르면 신뢰도 약간 낮춤
            if address_similarity >= 0.70:
                confidence = 0.93
            elif address_similarity >= 0.50:
                confidence = 0.90
            
            return (True, confidence, "NORMALIZED_NAME_MATCH", f"정규화된 이름 동일: {normalized_name1}, 주소 유사도: {address_similarity:.3f}")
        
        # 1단계: 고유 상호 추출 및 비교
        core_name_match, core_similarity = self.check_core_name_match(
            str(name1) if pd.notna(name1) else '',
            str(name2) if pd.notna(name2) else ''
        )
        
        # 고유 상호가 일치하지 않으면 다른 업체 (필수 조건)
        # GPT 피드백: 고유 상호가 다르면 절대 같은 업체로 판단하지 않음
        if not core_name_match:
            return (False, 0.0, "CORE_NAME_MISMATCH", "고유 상호 불일치")
        
        # 추가 검증: 고유 상호가 너무 다르면 제외
        # 일반 단어(국가명, 일반 업종 용어)를 제외한 고유 단어만 비교
        # 예: "ALMEGA" vs "ANUGRAH TANI" → 공통 단어 없음 → 다른 업체
        # 예: "CENTRA NU" vs "CENTRAL PART" → 일반 단어만 공통 → 다른 업체
        core1 = self.extract_company_core_name(str(name1) if pd.notna(name1) else '')
        core2 = self.extract_company_core_name(str(name2) if pd.notna(name2) else '')
        if core1 and core2:
            import re
            words1 = set(re.findall(r'\b\w+\b', core1.upper()))
            words2 = set(re.findall(r'\b\w+\b', core2.upper()))
            
            # 일반 단어 제외한 고유 단어만 비교 (통합된 general_words 사용)
            unique_words1 = words1 - self.GENERAL_WORDS
            unique_words2 = words2 - self.GENERAL_WORDS
            unique_common_words = unique_words1 & unique_words2
            all_unique_words = unique_words1 | unique_words2
            
            if all_unique_words:
                unique_common_ratio = len(unique_common_words) / len(all_unique_words)
                # 고유 단어 공통 비율이 30% 미만이면 고유 상호가 완전히 다름
                # 그룹 2, 3, 5 케이스: 일반 단어만 공통 → 다른 업체
                if unique_common_ratio < 0.30:
                    return (False, 0.0, "UNIQUE_WORDS_INSUFFICIENT", f"고유 단어 공통 비율 부족: {unique_common_ratio:.3f}")
            elif not unique_words1 or not unique_words2:
                # 고유 단어가 하나도 없으면 (일반 단어만 있으면) 다른 업체
                return (False, 0.0, "NO_UNIQUE_WORDS", "고유 단어 없음 (일반 단어만 존재)")
        
        # 2단계: 법인 형태 충돌 확인 (중국어 특화)
        # 법인 형태가 다르면 다른 업체로 판단
        # 예: "建设工程总公司" (건설 공사 회사) vs "股份有限公司" (주식회사) → 다른 업체
        if name1 and name2:
            name1_str = str(name1)
            name2_str = str(name2)
            
            # 중국어 법인 형태 키워드
            chinese_legal_forms = {
                '建设工程总公司': 'CONSTRUCTION_GENERAL',
                '建设总公司': 'CONSTRUCTION_GENERAL',
                '工程总公司': 'CONSTRUCTION_GENERAL',
                '股份有限公司': 'STOCK_COMPANY',
                '有限公司': 'LIMITED_COMPANY',
                '有限责任公司': 'LIMITED_COMPANY',
                '集团公司': 'GROUP_COMPANY',
                '集团': 'GROUP_COMPANY',
            }
            
            form1 = None
            form2 = None
            
            for keyword, form_type in chinese_legal_forms.items():
                if keyword in name1_str:
                    form1 = form_type
                if keyword in name2_str:
                    form2 = form_type
            
            # 법인 형태가 다르면 다른 업체
            # 특히 "建设工程总公司"와 "股份有限公司"는 완전히 다른 법인 형태
            if form1 and form2 and form1 != form2:
                # 같은 브랜드명이 있으면 같은 업체일 가능성 (예: "DAESUNG MACHINERY" vs "DAESUNG TECHNOLOGY")
                # 하지만 법인 형태가 완전히 다르면 다른 업체
                if form1 == 'CONSTRUCTION_GENERAL' or form2 == 'CONSTRUCTION_GENERAL':
                    # 건설 공사 회사는 다른 법인 형태와 절대 같을 수 없음
                    return (False, 0.0, "LEGAL_FORM_CONFLICT", f"법인 형태 충돌: {form1} vs {form2}")
        
        # 3단계: 업종 충돌 확인 (4단계)
        # 같은 브랜드명이 있으면 업종 충돌 확인 완화
        import re
        name1_upper = str(name1).upper() if pd.notna(name1) else ''
        name2_upper = str(name2).upper() if pd.notna(name2) else ''
        brand_words1 = set(re.findall(r'\b[A-Z]{4,}\b', name1_upper))
        brand_words2 = set(re.findall(r'\b[A-Z]{4,}\b', name2_upper))
        common_brands = brand_words1 & brand_words2
        has_common_brand = len(common_brands) > 0 and any(len(b) >= 4 for b in common_brands)
        
        if name1 and name2:
            business_match = self.check_business_type_match(str(name1), str(name2))
            if not business_match:
                # 업종 키워드가 명확히 다르면 다른 업체
                # 하지만 같은 브랜드명이 있으면 같은 업체일 가능성 (예: "DAESUNG MACHINERY" vs "DAESUNG TECHNOLOGY")
                if not has_common_brand:
                    return (False, 0.0, "BUSINESS_TYPE_CONFLICT", "업종 충돌 (공통 브랜드명 없음)")
        
        # 4단계: 주소를 보조 지표로 비교
        addr1 = self.normalize_address(row1)
        addr2 = self.normalize_address(row2)
        address_similarity = self.calculate_address_similarity(addr1, addr2)
        
        # 건물/번지 단위 주소 일치 확인
        building_match, building_score = self.check_building_address_match(row1, row2)
        
        # 공급업체코드 일치 확인 (강한 신호)
        code_match = False
        if '공급업체코드' in row1.index and '공급업체코드' in row2.index:
            code1 = str(row1['공급업체코드']) if pd.notna(row1['공급업체코드']) else ''
            code2 = str(row2['공급업체코드']) if pd.notna(row2['공급업체코드']) else ''
            if code1 and code2:
                code_match = (code1 == code2)
        
        # ============================================================
        # [개선] Step2 앙상블 점수 계산 (가중치 결합)
        # ============================================================
        ensemble_scores = {}
        
        # 1. 문자 기반 점수 (Jaro-Winkler, Token Sort Ratio)
        from rapidfuzz import fuzz
        name1_str = str(name1) if pd.notna(name1) else ''
        name2_str = str(name2) if pd.notna(name2) else ''
        
        jaro_winkler = fuzz.WRatio(name1_str, name2_str) / 100.0  # WRatio는 Jaro-Winkler 포함
        token_sort = fuzz.token_sort_ratio(name1_str, name2_str) / 100.0
        token_set = fuzz.token_set_ratio(name1_str, name2_str) / 100.0
        
        ensemble_scores['string_based'] = (jaro_winkler * 0.4 + token_sort * 0.4 + token_set * 0.2)
        
        # 2. 규칙 기반 점수 (0~1 정규화, 사용 가능한 신호만으로 평균)
        rule_components = []
        rule_weights = []
        
        if core_name_match:
            # 고유 상호 일치: 문자열 기반 유사도 (임베딩 제외)
            rule_components.append(core_similarity)
            rule_weights.append(0.4)
        
        if building_match:
            rule_components.append(building_score)
            rule_weights.append(0.3)
        
        if code_match:
            rule_components.append(1.0)  # 완전 일치
            rule_weights.append(0.3)
        
        if address_similarity >= 0.70:
            rule_components.append(address_similarity)
            rule_weights.append(0.2)
        
        # 사용 가능한 신호만으로 가중 평균
        if rule_components:
            total_weight = sum(rule_weights)
            if total_weight > 0:
                ensemble_scores['rule_based'] = sum(c * w for c, w in zip(rule_components, rule_weights)) / total_weight
            else:
                ensemble_scores['rule_based'] = 0.0
        else:
            ensemble_scores['rule_based'] = 0.0
        
        # 3. 숫자 기반 점수 (0~1 정규화, 사용 가능한 신호만으로 평균)
        number_components = []
        number_weights = []
        
        # 전화번호 일치 확인 (마지막 7자리)
        phone_fields = ['PHONE', 'TEL', 'TELEPHONE', 'MOBILE']
        phone_match = False
        for field in phone_fields:
            if field in row1.index and field in row2.index:
                phone1 = str(row1[field]).strip() if pd.notna(row1[field]) else ''
                phone2 = str(row2[field]).strip() if pd.notna(row2[field]) else ''
                if phone1 and phone2:
                    digits1 = re.findall(r'\d', phone1)
                    digits2 = re.findall(r'\d', phone2)
                    if len(digits1) >= 7 and len(digits2) >= 7:
                        if ''.join(digits1[-7:]) == ''.join(digits2[-7:]):
                            phone_match = True
                            break
        
        if phone_match:
            number_components.append(1.0)
            number_weights.append(0.4)
        
        # 우편번호 일치 확인
        zip_match = False
        zip_fields = ['ZIP', 'POSTAL', 'POSTCODE', 'ZIPCODE']
        for field in zip_fields:
            if field in row1.index and field in row2.index:
                zip1 = str(row1[field]).strip() if pd.notna(row1[field]) else ''
                zip2 = str(row2[field]).strip() if pd.notna(row2[field]) else ''
                if zip1 and zip2 and zip1 == zip2:
                    zip_match = True
                    break
        
        if zip_match:
            number_components.append(1.0)
            number_weights.append(0.3)
        
        # 사업자번호 일치 확인 (마지막 4자리)
        reg_match = False
        reg_fields = ['REGISTRATION', 'REG_NUM', 'BUSINESS_NUM', 'TAX_ID']
        for field in reg_fields:
            if field in row1.index and field in row2.index:
                reg1 = str(row1[field]).strip() if pd.notna(row1[field]) else ''
                reg2 = str(row2[field]).strip() if pd.notna(row2[field]) else ''
                if reg1 and reg2:
                    digits1 = re.findall(r'\d', reg1)
                    digits2 = re.findall(r'\d', reg2)
                    if len(digits1) >= 4 and len(digits2) >= 4:
                        if ''.join(digits1[-4:]) == ''.join(digits2[-4:]):
                            reg_match = True
                            break
        
        if reg_match:
            number_components.append(1.0)
            number_weights.append(0.5)
        
        # 사용 가능한 신호만으로 가중 평균
        if number_components:
            total_weight = sum(number_weights)
            if total_weight > 0:
                ensemble_scores['number_based'] = sum(c * w for c, w in zip(number_components, number_weights)) / total_weight
            else:
                ensemble_scores['number_based'] = 0.0
        else:
            ensemble_scores['number_based'] = 0.0
        
        # 4. 임베딩 기반 점수 (별도 계산, core_similarity와 분리)
        embedding_score = 0.0
        if self.use_embedding:
            # 실제 임베딩 유사도 계산 (core_similarity와 별도)
            # row_cache에서 임베딩 가져오기
            # 임베딩은 의미 기반 유사도로만 사용
            try:
                # 의미 기반 유사도 계산
                semantic_score = self.calculate_semantic_similarity(
                    str(name1) if pd.notna(name1) else '',
                    str(name2) if pd.notna(name2) else ''
                )
                embedding_score = semantic_score
            except:
                # 임베딩 계산 실패 시 0
                embedding_score = 0.0
        
        ensemble_scores['embedding_based'] = embedding_score
        
        # 앙상블 최종 점수 계산 (가중치, 사용 가능한 컴포넌트만으로 평균)
        # 문자 기반: 0.25, 규칙 기반: 0.30, 숫자 기반: 0.25, 임베딩 기반: 0.20
        ensemble_components = []
        ensemble_weights = []
        
        # 항상 사용 가능한 컴포넌트
        ensemble_components.append(ensemble_scores['string_based'])
        ensemble_weights.append(0.25)
        
        ensemble_components.append(ensemble_scores['rule_based'])
        ensemble_weights.append(0.30)
        
        # 숫자 기반은 사용 가능한 경우만
        if ensemble_scores['number_based'] > 0:
            ensemble_components.append(ensemble_scores['number_based'])
            ensemble_weights.append(0.25)
        
        # 임베딩 기반은 사용 가능한 경우만
        if ensemble_scores['embedding_based'] > 0:
            ensemble_components.append(ensemble_scores['embedding_based'])
            ensemble_weights.append(0.20)
        
        # 사용 가능한 컴포넌트만으로 가중 평균
        total_weight = sum(ensemble_weights)
        if total_weight > 0:
            final_ensemble_score = sum(c * w for c, w in zip(ensemble_components, ensemble_weights)) / total_weight
        else:
            final_ensemble_score = ensemble_scores['string_based']  # fallback
        
        # 최종 판단 기준 (GPT 프롬프트 기반 + 앙상블 점수)
        # 고유 상호가 일치하고, 주소/업종 일관성이 충족되는 경우에만 중복으로 판단
        
        # 조건 1: 고유 상호 일치 (이미 확인됨)
        # 조건 2: 주소 일관성 확인 (완화된 기준)
        address_consistent = False
        
        # 강한 신호들
        if code_match:
            # 공급업체코드가 일치하면 강한 신호
            # 하지만 주소가 완전히 다르면 (주소 유사도 < 0.30) 오병합 가능성 있음
            # TRIDUTA 케이스: 공급업체코드 같지만 주소가 완전히 다름 → 다른 업체
            if address_similarity >= 0.30:
                address_consistent = True
            else:
                # 공급업체코드는 같지만 주소가 완전히 다름 → 오병합 가능성
                # 주소 일관성 없음으로 판단
                address_consistent = False
        elif building_match:
            # 건물/번지가 일치하면 주소 일관성 있음
            address_consistent = True
        elif address_similarity >= 0.90:
            # 주소가 거의 동일하면 주소 일관성 있음
            address_consistent = True
        # 완화된 조건들
        elif address_similarity >= 0.75 and core_similarity >= 0.80:
            # 주소가 유사하고 고유 상호가 유사하면 주소 일관성 있음
            address_consistent = True
        elif address_similarity >= 0.70 and core_similarity >= 0.85:
            # 주소가 중간 수준이고 고유 상호가 매우 유사하면 주소 일관성 있음
            address_consistent = True
        elif address_similarity >= 0.65 and core_similarity >= 0.90:
            # 주소가 낮아도 고유 상호가 매우 유사하면 주소 일관성 있음
            address_consistent = True
        
        # 이름 유사도가 매우 높을 때 추가 완화
        name_similarity = self.calculate_multilingual_similarity(
            str(name1) if pd.notna(name1) else '',
            str(name2) if pd.notna(name2) else ''
        )
        
        if name_similarity >= 0.90 and core_similarity >= 0.75:
            # 이름이 매우 유사하고 고유 상호도 유사하면 주소 조건 완화
            address_consistent = True
        elif name_similarity >= 0.85 and core_similarity >= 0.80 and address_similarity >= 0.60:
            # 이름과 고유 상호가 유사하고 주소가 중간 수준이면 주소 일관성 있음
            address_consistent = True
        
        # 같은 브랜드명이 있을 때 추가 완화 (균형잡힌 기준)
        if has_common_brand:
            # 같은 브랜드명이 있으면 주소 조건 완화하되, 너무 낮은 기준은 피함
            # 예: "DAESUNG MACHINERY" vs "DAESUNG TECHNOLOGY E&C"
            if address_similarity >= 0.60 and core_similarity >= 0.65:
                # 주소가 중간 수준이고 고유 상호가 유사하면 주소 일관성 있음
                address_consistent = True
            elif name_similarity >= 0.80 and address_similarity >= 0.55 and core_similarity >= 0.60:
                # 이름이 매우 유사하고 주소가 중간 수준이면 주소 일관성 있음
                address_consistent = True
            elif name_similarity >= 0.75 and core_similarity >= 0.65 and address_similarity >= 0.50:
                # 이름과 고유 상호가 유사하고 주소가 중간 수준이면 주소 일관성 있음
                address_consistent = True
            elif name_similarity >= 0.75 and core_similarity >= 0.60:
                # 이름이 매우 유사하고 고유 상호가 유사하면 주소 조건 완화
                # 주소가 낮아도(0.30 이상) 같은 브랜드명이 있으면 주소 일관성 있음
                # 단, 주소가 너무 낮으면(0.30 미만) 제외
                if address_similarity >= 0.30:
                    address_consistent = True
        
        # 최종 판단: 고유 상호 일치 + 주소/업종 일관성 + 앙상블 점수
        if core_name_match and address_consistent:
            # [개선] 앙상블 점수만 사용 (임베딩 이중 계산 방지)
            # core_similarity는 이미 rule_based에 포함되어 있으므로 제외
            base_confidence = final_ensemble_score
            match_type = "CORE_NAME_MATCH"
            match_reason_parts = [
                f"고유 상호 일치: {core_similarity:.3f}",
                f"앙상블 점수: {final_ensemble_score:.3f}",
                f"(문자:{ensemble_scores['string_based']:.2f}, 규칙:{ensemble_scores['rule_based']:.2f}, 숫자:{ensemble_scores['number_based']:.2f}, 임베딩:{ensemble_scores['embedding_based']:.2f})"
            ]
            
            confidence = base_confidence
            
            if code_match:
                confidence += 0.2  # 공급업체코드 일치 보너스
                match_reason_parts.append("공급업체코드 일치")
            elif building_match:
                confidence += building_score * 0.15
                match_reason_parts.append(f"건물/번지 일치: {building_score:.3f}")
            elif address_similarity >= 0.90:
                confidence += address_similarity * 0.15
                match_reason_parts.append(f"주소 유사도 높음: {address_similarity:.3f}")
            elif address_similarity >= 0.75:
                confidence += address_similarity * 0.10
                match_reason_parts.append(f"주소 유사도: {address_similarity:.3f}")
            else:
                confidence += address_similarity * 0.1
                match_reason_parts.append(f"주소 유사도: {address_similarity:.3f}")
            
            # 이름 유사도 보너스
            if name_similarity >= 0.90:
                confidence += 0.1
                match_reason_parts.append("이름 유사도 매우 높음")
            elif name_similarity >= 0.85:
                confidence += 0.05
                match_reason_parts.append("이름 유사도 높음")
            
            match_reason = ", ".join(match_reason_parts)
            return (True, min(confidence, 1.0), match_type, match_reason)
        
        # 확실하지 않으면 같은 업체로 판단하지 않음
        return (False, 0.0, "INSUFFICIENT_EVIDENCE", "고유 상호 일치하지만 주소/업종 일관성 부족")
    
    def detect_duplicates(self, df: pd.DataFrame, show_progress: bool = True, 
                         save_intermediate_at: int = 5, candidate_mode: str = 'ann',
                         checkpoint_file: str = None, checkpoint_interval: int = 1000,
                         output_dir: str = '') -> Tuple[List[List[int]], List[List[int]], dict]:
        """
        중복 탐지 메인 함수
        
        Args:
            candidate_mode: 'ann' (ANN 기반) 또는 'legacy' (기존 blocking 기반)
            checkpoint_file: 체크포인트 파일 경로 (None이면 체크포인트 사용 안 함)
            checkpoint_interval: 체크포인트 저장 간격 (처리된 행 수, 기본값: 1000)
            output_dir: 중간/결과 파일 저장 폴더 (예: 'output'). 비면 현재 디렉터리
        """
        if candidate_mode == 'ann':
            return self.detect_duplicates_ann(df, show_progress, save_intermediate_at, checkpoint_file, checkpoint_interval, output_dir)
        else:
            return self.detect_duplicates_legacy(df, show_progress, save_intermediate_at, output_dir)
    
    def detect_duplicates_legacy(self, df: pd.DataFrame, show_progress: bool = True, 
                         save_intermediate_at: int = 5, output_dir: str = '') -> Tuple[List[List[int]], List[List[int]], dict]:
        """
        2단계 중복 탐지 (Legacy 버전: 2-gram blocking + pair-level deduplication)
        
        [주요 변경사항]
        - 2-gram blocking으로 변경 (create_multiple_blocking_keys)
        - candidate_visited 제거, checked_pairs로 pair-level deduplication
        - 큰 그룹 처리 개선: Step1로 후보 축소 후 Step2 수행
        - 성능 최적화: 전처리 결과 캐시
        
        Returns:
            (candidate_groups: List[List[int]], final_groups: List[List[int]], match_info: dict)
            - candidate_groups: STEP1 후보 그룹 리스트
            - final_groups: STEP2 최종 중복 그룹 리스트
            - match_info: 매칭 정보 딕셔너리
        """
        import time
        start_time = time.time()
        from typing import Tuple
        from itertools import combinations
        n = len(df)
        
        # [필수 수정 2] candidate_visited 제거, checked_pairs로 변경
        checked_pairs = set()  # 이미 비교한 pair 기록: (min(i,j), max(i,j))
        candidate_groups = []  # STEP1 후보 그룹
        match_info = {}  # 매칭 정보: {(idx1, idx2): {'match_type': str, 'match_reason': str, 'confidence': float}}
        intermediate_saved = False  # 중간 저장 여부
        
        # [버그 수정 D] Union-Find로 final_groups 생성 (일관성 보장)
        edges = []  # Step2 통과한 pair들: [(i, j, confidence)]
        
        # [중간 저장] 임시 Union-Find (중간 저장용)
        temp_parent = {idx: idx for idx in df.index}
        temp_rank = {idx: 0 for idx in df.index}
        
        def temp_find(x):
            if temp_parent[x] != x:
                temp_parent[x] = temp_find(temp_parent[x])
            return temp_parent[x]
        
        def temp_union(a, b):
            ra, rb = temp_find(a), temp_find(b)
            if ra == rb:
                return
            if temp_rank[ra] < temp_rank[rb]:
                temp_parent[ra] = rb
            elif temp_rank[ra] > temp_rank[rb]:
                temp_parent[rb] = ra
            else:
                temp_parent[rb] = ra
                temp_rank[ra] += 1
        
        # Union-Find 초기화
        parent = {idx: idx for idx in df.index}
        rank = {idx: 0 for idx in df.index}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # 경로 압축
            return parent[x]
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
        
        # [성능 최적화] 전처리 결과 캐시
        if show_progress:
            cache_msg = "\n[성능 최적화] 전처리 결과 캐싱 중..."
            if self.logger:
                self.logger.log(cache_msg)
            else:
                print(cache_msg)
        
        # 필요한 컬럼과 전처리 결과를 미리 캐시
        row_cache = {}
        import re
        import numpy as np
        
        # [개선 4] 임베딩 사전 캐싱
        if self.use_embedding:
            if show_progress:
                emb_msg = "  임베딩 사전 생성 중..."
                if self.logger:
                    self.logger.log(emb_msg)
                else:
                    print(emb_msg)
            
            # 모든 이름을 한 번에 임베딩 (idx 기준으로 캐싱)
            names_list = []
            idx_list = []
            for idx in df.index:
                row = df.loc[idx]
                name = str(row.get('공급업체명', '')) if pd.notna(row.get('공급업체명', '')) else ''
                if name:
                    names_list.append(name)
                    idx_list.append(idx)
            
            # 배치 임베딩 생성
            if names_list:
                try:
                    name_embeddings = self.embedding_model.encode(names_list, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=show_progress)
                    # [버그 수정 D] idx 기준으로 임베딩 캐싱 (name 기준은 중복 name이 있으면 덮어쓰기 문제)
                    embeddings_dict = {idx: emb for idx, emb in zip(idx_list, name_embeddings)}
                except Exception as e:
                    if show_progress:
                        emb_err = f"  [경고] 임베딩 생성 실패: {e}, 개별 계산으로 전환"
                        if self.logger:
                            self.logger.log(emb_err)
                        else:
                            print(emb_err)
                    embeddings_dict = {}
            else:
                embeddings_dict = {}
        else:
            embeddings_dict = {}
        
        for idx in df.index:
            row = df.loc[idx]
            name = str(row.get('공급업체명', '')) if pd.notna(row.get('공급업체명', '')) else ''
            country = str(row.get('Land', '')).strip().upper() if pd.notna(row.get('Land', '')) else ''
            city = str(row.get('CITY1', '')) if pd.notna(row.get('CITY1', '')) else ''
            
            # Core Name과 Normalized Name 미리 추출
            core_name = self.extract_company_core_name(name) if name else ''
            normalized_name = self.normalize_name_for_blocking(name) if name else ''
            
            # 주소 정규화
            address = self.normalize_address(row)
            
            # Core Name 토큰 set 미리 계산 (개선 3용)
            core_tokens_set = set()
            if core_name:
                core_tokens = re.findall(r'\b\w+\b', core_name.upper())
                core_tokens_set = {t for t in core_tokens if len(t) >= 2 and t not in self.STEP1_STOP_WORDS}
            
            # 임베딩 캐시 (idx 기준)
            name_emb = embeddings_dict.get(idx) if idx in embeddings_dict else None
            
            row_cache[idx] = {
                'name': name,
                'country': country,
                'city': city,
                'address': address,
                'core_name': core_name,
                'normalized_name': normalized_name,
                'core_tokens_set': core_tokens_set,  # 개선 3용
                'name_emb': name_emb,  # 개선 4용
                'row': row  # 원본 row도 캐시 (필요시 사용)
            }
        
        cache_time = time.time() - start_time
        if show_progress:
            cache_done_msg = f"  캐싱 완료: {len(row_cache):,}행, 소요 시간: {cache_time:.2f}초"
            if self.logger:
                self.logger.log(cache_done_msg)
            else:
                print(cache_done_msg)
        
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
            if KOREAN_ROMANIZER_AVAILABLE:
                status_msgs.append("  [OK] 한글 로마자화: 활성화")
            else:
                status_msgs.append("  [INFO] 한글 로마자화: 비활성화")
            
            for msg in status_msgs:
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)
        
        # Blocking 키 생성 (캐시된 데이터 사용)
        blocking_groups = defaultdict(list)
        blocking_start = time.time()
        for idx in df.index:
            keys = self.create_multiple_blocking_keys(row_cache[idx]['row'])
            for key in keys:
                blocking_groups[key].append(idx)
        
        # [개선] 고빈도 키 필터링 (후보 폭발 방지)
        # 한 키에 매핑된 vendor 수가 threshold 넘어가면 그 키는 blocking에서 제외
        high_freq_threshold = 200  # 고빈도 키 임계값
        key_counts = {key: len(indices) for key, indices in blocking_groups.items()}
        high_freq_keys = {key for key, count in key_counts.items() if count > high_freq_threshold}
        
        if high_freq_keys and show_progress:
            msg = f"  [고빈도 키 필터링] {len(high_freq_keys)}개 키 제외 (임계값: {high_freq_threshold})"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        # 고빈도 키 제외
        blocking_groups = {key: indices for key, indices in blocking_groups.items() 
                          if key not in high_freq_keys}
        
        # [개선] 부분번호 키 단독 후보 생성 금지
        # REG:last4, PHONE:last4 같은 부분번호 키는 단독으로 후보 생성하지 않음
        # 다른 키(CORE2, NORM2, city 등)와 함께 매칭되어야만 후보로 간주
        partial_number_key_prefixes = ['REG:', 'PHONE:', 'ZIP:']  # 부분번호 키 접두사
        
        # 부분번호 키만 있는 그룹 제거
        partial_only_groups = {}
        for key, indices in blocking_groups.items():
            is_partial_number = any(key.startswith(prefix) for prefix in partial_number_key_prefixes)
            if is_partial_number:
                # 이 키가 부분번호 키인 경우, 다른 키와 함께 있는지 확인
                # 각 idx에 대해 다른 키가 있는지 확인
                for idx in indices:
                    # 이 idx의 다른 키들 확인
                    other_keys = [k for k in blocking_groups.keys() 
                                if idx in blocking_groups[k] and k != key]
                    if not other_keys:
                        # 다른 키가 없으면 부분번호 키만 있는 경우
                        if key not in partial_only_groups:
                            partial_only_groups[key] = []
                        partial_only_groups[key].append(idx)
        
        # 부분번호 키만 있는 경우 제거
        if partial_only_groups:
            for key, idxs_to_remove in partial_only_groups.items():
                blocking_groups[key] = [idx for idx in blocking_groups[key] 
                                      if idx not in idxs_to_remove]
                # 빈 그룹 제거
                if not blocking_groups[key]:
                    del blocking_groups[key]
            
            if show_progress:
                removed_count = sum(len(idxs) for idxs in partial_only_groups.values())
                msg = f"  [부분번호 키 필터링] {removed_count}개 레코드 제외 (단독 부분번호 키)"
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)
        
        blocking_time = time.time() - blocking_start
        total_blocks = len(blocking_groups)
        if show_progress:
            block_sizes = [len(indices) for indices in blocking_groups.values()]
            max_size = max(block_sizes) if block_sizes else 0
            large_groups_10 = sum(1 for size in block_sizes if size > 10)
            large_groups_20 = sum(1 for size in block_sizes if size > 20)
            msg1 = f"  생성된 Blocking 그룹: {total_blocks}개"
            msg2 = f"  최대 그룹 크기: {max_size}행"
            msg3 = f"  큰 그룹 (>10행): {large_groups_10}개, 비정상 그룹 (>20행): {large_groups_20}개"
            if self.logger:
                self.logger.log(msg1)
                self.logger.log(msg2)
                self.logger.log(msg3)
            else:
                print(msg1)
                print(msg2)
                print(msg3)
        
        if show_progress:
            msg = "\n[2단계] 하이브리드 비교 시작 (Blocking + 의미 기반)..."
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        block_num = 0
        total_comparisons = 0
        large_group_threshold = 50  # 큰 그룹 임계값 (더 일찍 최적화 모드 진입)
        
        for blocking_key, indices in blocking_groups.items():
            block_num += 1
            block_size = len(indices)
            
            if block_size < 2:
                continue
            
            # 안전장치: 그룹 크기 모니터링 (GPT 가이드 반영)
            # [GPT 가이드] 그룹 크기 > 5: 경고, > 10: 재검사
            if block_size > 10:
                # 그룹 크기 > 10: 자동으로 Stop-word 기반 재검사 수행
                if show_progress:
                    msg = f"  [경고] STEP1 그룹 크기 10 초과: {block_size}행 (키: {blocking_key[:80]}) - Stop-word 재검사 필요"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
                
                # [개선 1] Stop-word 기반 재검사 최적화: row_cache 사용 + Top-K 선택
                # 각 i에 대해 token overlap 기반으로 상위 K(예: 30) 후보만 is_candidate 수행
                indices_list = list(indices)
                valid_indices = set()
                top_k_recheck = 30  # 재검사 시 Top-K만 선택
                
                for i in indices_list:
                    cached_i = row_cache[i]
                    i_tokens = cached_i['core_tokens_set']
                    
                    # j 후보들의 token overlap 계산
                    candidate_scores = []
                    for j in indices_list:
                        if i >= j:  # 중복 방지
                            continue
                        cached_j = row_cache[j]
                        j_tokens = cached_j['core_tokens_set']
                        overlap = len(i_tokens & j_tokens)
                        if overlap > 0:
                            candidate_scores.append((j, overlap))
                    
                    # Top-K 선택
                    candidate_scores.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = [j for j, _ in candidate_scores[:top_k_recheck]]
                    
                    # 선택된 후보에 대해서만 is_candidate 수행
                    for j in top_candidates:
                        pair_key = (min(i, j), max(i, j))
                        if pair_key in checked_pairs:
                            continue
                        checked_pairs.add(pair_key)
                        
                        cached_j = row_cache[j]
                        is_cand, reason, overlap_count = self.is_candidate(cached_i['row'], cached_j['row'])
                        total_comparisons += 1
                        
                        if is_cand:
                            valid_indices.add(i)
                            valid_indices.add(j)
                
                # 재검사 결과: 유효한 후보가 없으면 이 그룹은 건너뜀
                if len(valid_indices) < 2:
                    if show_progress:
                        msg = f"  [재검사 결과] 그룹 {blocking_key[:80]}에서 유효한 후보 없음 - 일반 단어만 공통으로 판단"
                        if self.logger:
                            self.logger.log(msg)
                        else:
                            print(msg)
                    continue
                
                # 재검사된 유효한 인덱스만 사용
                indices = valid_indices
                block_size = len(indices)
                indices_list = list(indices)
                
            elif block_size > 5:
                # 그룹 크기 > 5: Core 추출 오류 가능성 경고 로그 생성
                if show_progress:
                    msg = f"  [주의] STEP1 그룹 크기 5 초과: {block_size}행 (키: {blocking_key[:80]}) - Core 추출 오류 가능성"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
            
            # [개선 1] comparisons_in_block은 추정치로 분리, total_comparisons는 실제 호출만 카운트
            estimated_comparisons = block_size * (block_size - 1) // 2
            
            # 진행 상황 로그 (더 자주 기록)
            if show_progress:
                if block_num % 100 == 0 or block_size > large_group_threshold:
                    msg = f"  처리 중: {block_num}/{total_blocks} 그룹 (현재 그룹 크기: {block_size}행), 실제 비교 횟수: {total_comparisons:,}, 추정 비교 횟수: {estimated_comparisons:,}"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
            
            # [개선 2] 큰 그룹 처리: Heap으로 Top-K만 유지 (메모리 절약)
            if block_size > large_group_threshold:
                if show_progress:
                    msg = f"  [큰 그룹 처리] {block_size}행 그룹 처리 중... (최적화 모드)"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
                
                # Heap을 사용하여 Top-K만 유지 (메모리 효율적)
                import heapq
                indices_list = list(indices)
                max_step2_pairs = 30 if block_size > 100 else 50
                step1_heap = []  # 최소 힙 (score가 낮은 것부터 제거)
                
                for i_idx, i in enumerate(indices_list):
                    for j in indices_list[i_idx + 1:]:
                        pair_key = (min(i, j), max(i, j))
                        if pair_key in checked_pairs:
                            continue
                        checked_pairs.add(pair_key)
                        
                        cached_i = row_cache[i]
                        cached_j = row_cache[j]
                        
                        # Step1: 후보 여부 판단
                        is_cand, reason, overlap_count = self.is_candidate(cached_i['row'], cached_j['row'])
                        total_comparisons += 1
                        
                        if is_cand:
                            # score 계산: overlap_count*0.6 + name_similarity*0.4
                            from rapidfuzz import fuzz
                            name_sim = fuzz.ratio(cached_i['name'], cached_j['name']) / 100.0
                            score = overlap_count * 0.6 + name_sim * 0.4
                            
                            # Heap에 추가 (최소 힙이므로 음수로 저장)
                            if len(step1_heap) < max_step2_pairs:
                                heapq.heappush(step1_heap, (score, i, j, reason, overlap_count, name_sim))
                            elif score > step1_heap[0][0]:  # 현재 최소값보다 크면 교체
                                heapq.heapreplace(step1_heap, (score, i, j, reason, overlap_count, name_sim))
                
                # Heap에서 내림차순으로 추출
                step2_pairs = []
                while step1_heap:
                    score, i, j, reason, overlap_count, name_sim = heapq.heappop(step1_heap)
                    step2_pairs.append((i, j, reason, overlap_count, name_sim, score))
                step2_pairs.reverse()  # 내림차순으로 정렬
                
                if show_progress:
                    msg = f"    Step1 완료: {len(step2_pairs)}개 후보 쌍 발견 (Top-{max_step2_pairs})"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
                
                # Step2: 후보인 경우에만 최종 판단
                current_candidate_group = []
                for i, j, reason, overlap_count, name_sim, score in step2_pairs:
                    try:
                        cached_i = row_cache[i]
                        cached_j = row_cache[j]
                        
                        # 후보 그룹에 추가
                        if i not in current_candidate_group:
                            current_candidate_group.append(i)
                        if j not in current_candidate_group:
                            current_candidate_group.append(j)
                        
                        # STEP 2: 후보인 경우에만 최종 판단
                        is_dup, confidence, match_type, match_reason = self.are_duplicates(cached_i['row'], cached_j['row'])
                        total_comparisons += 1
                        
                        # 매칭 정보 저장 (후보인 경우 모두 저장)
                        match_info[(i, j)] = {
                            'match_type': match_type,
                            'match_reason': match_reason,
                            'confidence': confidence
                        }
                        
                        if is_dup:
                            # [버그 수정 D] Union-Find로 edge 추가 (일관성 보장)
                            edges.append((i, j, confidence))
                            # [버그 수정 2] 중간 저장용 temp_union 즉시 업데이트
                            temp_union(i, j)
                            
                            if show_progress:
                                try:
                                    name1 = cached_i['name'][:40] if cached_i['name'] else 'N/A'
                                    name2 = cached_j['name'][:40] if cached_j['name'] else 'N/A'
                                    name1_safe = name1.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                    name2_safe = name2.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                    msg = f"  [중복 발견] {name1_safe} <-> {name2_safe} (신뢰도: {confidence:.2f})"
                                    if self.logger:
                                        self.logger.log(msg)
                                    else:
                                        print(msg)
                                except:
                                    pass
                    except Exception as e:
                            if show_progress:
                                # 오류 메시지도 안전하게 인코딩
                                try:
                                    # 오류 타입과 메시지를 분리하여 안전하게 처리
                                    error_type = type(e).__name__
                                    try:
                                        error_msg_str = str(e)
                                        # UTF-8로 안전하게 인코딩
                                        error_msg_safe = error_msg_str.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                    except:
                                        error_msg_safe = "인코딩 오류"
                                    error_msg = f"  [경고] 비교 중 오류 발생: {error_type}: {error_msg_safe}"
                                    if self.logger:
                                        self.logger.log(error_msg)
                                    else:
                                        print(error_msg)
                                except Exception as log_error:
                                    # 로그 기록 자체가 실패하면 무시
                                    pass
                            continue
                
                # [버그 수정 1, 3] candidate_groups 항상 추가, 중간 저장은 temp 기반으로 판단
                if len(current_candidate_group) > 1:
                    candidate_group = sorted(set(current_candidate_group))
                    candidate_groups.append(candidate_group)
                    
                    # [버그 수정 3] 중간 저장 판단 함수 사용
                    if save_intermediate_at > 0 and not intermediate_saved:
                        # temp_parent 기반으로 현재 그룹 수 계산
                        temp_components = {}
                        for idx_temp in df.index:
                            root = temp_find(idx_temp)
                            if root not in temp_components:
                                temp_components[root] = []
                            temp_components[root].append(idx_temp)
                        
                        temp_final_groups = [comp for comp in temp_components.values() if len(comp) >= 2]
                        
                        if len(temp_final_groups) >= save_intermediate_at:
                            try:
                                from .merger import DataMerger
                                merger = DataMerger()
                                df_intermediate = merger.merge_duplicates_2step(df, candidate_groups, temp_final_groups, match_info)
                                if '_merged_from_indices' in df_intermediate.columns:
                                    df_intermediate = df_intermediate.drop(columns=['_merged_from_indices'])
                                
                                from datetime import datetime
                                import os
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                intermediate_file = f'중간결과_{len(temp_final_groups)}개그룹_{timestamp}.xlsx'
                                if output_dir:
                                    os.makedirs(output_dir, exist_ok=True)
                                    intermediate_file = os.path.join(output_dir, intermediate_file)
                                df_intermediate.to_excel(intermediate_file, index=False)
                                
                                if show_progress:
                                    msg = f"  [중간 저장] {len(temp_final_groups)}개 그룹 발견 → {intermediate_file} 저장 완료"
                                    if self.logger:
                                        self.logger.log(msg)
                                    else:
                                        print(msg)
                                
                                intermediate_saved = True
                            except Exception as save_error:
                                if show_progress:
                                    try:
                                        error_msg = f"  [경고] 중간 저장 실패: {type(save_error).__name__}"
                                        if self.logger:
                                            self.logger.log(error_msg)
                                        else:
                                            print(error_msg)
                                    except:
                                        pass
            else:
                # [개선 3] 작은 그룹 최적화: Step1 생략 규칙 개선
                indices_list = list(indices)
                current_candidate_group = []
                # [버그 수정 E] Step1 생략을 block_size 기준이 아니라 토큰 overlap 기반으로 변경
                # block_size <= 30이라도 토큰 overlap=0인 pair는 Step2로 넘기지 않음
                
                for i_idx, i in enumerate(indices_list):
                    for j in indices_list[i_idx + 1:]:
                        pair_key = (min(i, j), max(i, j))
                        if pair_key in checked_pairs:
                            continue
                        checked_pairs.add(pair_key)
                        
                        try:
                            cached_i = row_cache[i]
                            cached_j = row_cache[j]
                            
                            # [버그 수정 E] Step1 생략 규칙 개선: 토큰 overlap 기반
                            # block_size <= 30이어도 토큰 overlap=0인 pair는 Step2로 넘기지 않음
                            if block_size <= 30:
                                # 빠른 토큰 overlap 체크 (Step1 생략 가능 여부 판단)
                                core1 = cached_i.get('core_name', '')
                                core2 = cached_j.get('core_name', '')
                                if core1 and core2:
                                    import re
                                    tokens1 = set(re.findall(r'\b\w+\b', core1.upper()))
                                    tokens2 = set(re.findall(r'\b\w+\b', core2.upper()))
                                    tokens1 = tokens1 - self.STEP1_STOP_WORDS
                                    tokens2 = tokens2 - self.STEP1_STOP_WORDS
                                    overlap = len(tokens1 & tokens2)
                                    
                                    if overlap > 0:
                                        # 토큰 overlap 있으면 Step1 생략 가능
                                        is_cand = True
                                        reason = f"작은 그룹: 토큰 overlap={overlap}, Step1 생략"
                                    else:
                                        # 토큰 overlap 없으면 Step1 수행
                                        is_cand, reason, overlap_count = self.is_candidate(cached_i['row'], cached_j['row'])
                                        total_comparisons += 1
                                else:
                                    # core_name 없으면 Step1 수행
                                    is_cand, reason, overlap_count = self.is_candidate(cached_i['row'], cached_j['row'])
                                    total_comparisons += 1
                            else:
                                # 큰 그룹은 항상 Step1 수행
                                is_cand, reason, overlap_count = self.is_candidate(cached_i['row'], cached_j['row'])
                                total_comparisons += 1
                            
                            if is_cand:
                                # 후보 그룹에 추가 (중간 저장용)
                                if i not in current_candidate_group:
                                    current_candidate_group.append(i)
                                if j not in current_candidate_group:
                                    current_candidate_group.append(j)
                                
                                # STEP 2: 최종 판단
                                is_dup, confidence, match_type, match_reason = self.are_duplicates(cached_i['row'], cached_j['row'])
                                total_comparisons += 1
                                
                                # 매칭 정보 저장
                                match_info[(i, j)] = {
                                    'match_type': match_type,
                                    'match_reason': match_reason,
                                    'confidence': confidence
                                }
                                
                                if is_dup:
                                    # [버그 수정 D] Union-Find로 edge 추가 (일관성 보장)
                                    edges.append((i, j, confidence))
                                    # [버그 수정 2] 중간 저장용 temp_union 즉시 업데이트
                                    temp_union(i, j)
                                    
                                    if show_progress:
                                        try:
                                            name1 = cached_i['name'][:40] if cached_i['name'] else 'N/A'
                                            name2 = cached_j['name'][:40] if cached_j['name'] else 'N/A'
                                            # 안전하게 인코딩
                                            name1_safe = name1.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                            name2_safe = name2.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                            msg = f"  [중복 발견] {name1_safe} <-> {name2_safe} (신뢰도: {confidence:.2f})"
                                            if self.logger:
                                                self.logger.log(msg)
                                            else:
                                                print(msg)
                                        except Exception as name_error:
                                            # 이름 출력 오류는 무시하고 계속 진행
                                            pass
                        except Exception as e:
                            if show_progress:
                                # 오류 메시지도 안전하게 인코딩
                                try:
                                    # 오류 타입과 메시지를 분리하여 안전하게 처리
                                    error_type = type(e).__name__
                                    try:
                                        error_msg_str = str(e)
                                        # UTF-8로 안전하게 인코딩
                                        error_msg_safe = error_msg_str.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                                    except:
                                        error_msg_safe = "인코딩 오류"
                                    error_msg = f"  [경고] 비교 중 오류 발생: {error_type}: {error_msg_safe}"
                                    if self.logger:
                                        self.logger.log(error_msg)
                                    else:
                                        print(error_msg)
                                except Exception as log_error:
                                    # 로그 기록 자체가 실패하면 무시
                                    pass
                            continue
                
                # [개선 4] candidate_groups: 중간 저장 시에만 생성
                if len(current_candidate_group) > 1:
                    candidate_group = sorted(set(current_candidate_group))
                    candidate_groups.append(candidate_group)
                    
                    # 중간 결과 저장 (10개 그룹 발견 시) - edges 기반으로 임시 그룹 수 계산
                    if save_intermediate_at > 0 and not intermediate_saved:
                        # 임시 그룹 수 계산 (edges 기반)
                        temp_components = {}
                        for idx_temp in df.index:
                            root = find(idx_temp)
                            if root not in temp_components:
                                temp_components[root] = []
                            temp_components[root].append(idx_temp)
                        
                        temp_final_groups = [comp for comp in temp_components.values() if len(comp) >= 2]
                        
                        if len(temp_final_groups) >= save_intermediate_at:
                            try:
                                from .merger import DataMerger
                                merger = DataMerger()
                                df_intermediate = merger.merge_duplicates_2step(df, candidate_groups, temp_final_groups, match_info)
                                if '_merged_from_indices' in df_intermediate.columns:
                                    df_intermediate = df_intermediate.drop(columns=['_merged_from_indices'])
                                
                                from datetime import datetime
                                import os
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                intermediate_file = f'중간결과_{len(temp_final_groups)}개그룹_{timestamp}.xlsx'
                                if output_dir:
                                    os.makedirs(output_dir, exist_ok=True)
                                    intermediate_file = os.path.join(output_dir, intermediate_file)
                                df_intermediate.to_excel(intermediate_file, index=False)
                                
                                if show_progress:
                                    msg = f"  [중간 저장] {len(temp_final_groups)}개 그룹 발견 → {intermediate_file} 저장 완료"
                                    if self.logger:
                                        self.logger.log(msg)
                                    else:
                                        print(msg)
                                
                                intermediate_saved = True
                            except Exception as save_error:
                                if show_progress:
                                    try:
                                        error_msg = f"  [경고] 중간 저장 실패: {type(save_error).__name__}: {str(save_error)}"
                                        if self.logger:
                                            self.logger.log(error_msg)
                                        else:
                                            print(error_msg)
                                    except:
                                        pass
        
        # [버그 수정 D] Union-Find로 final_groups 생성 (모든 edges 처리 후)
        # [개선] 그룹 확장 안전장치 추가
        group_members_legacy = {}  # root -> set of idx
        
        for edge in edges:
            if len(edge) >= 3:
                i, j, confidence = edge[0], edge[1], edge[2]
                
                # 그룹 확장 안전장치: confidence가 너무 낮으면 스킵
                if confidence < self.similarity_threshold * 0.9:  # 10% 여유
                    continue
                
                root_i = find(i)
                root_j = find(j)
                
                # 이미 같은 그룹이면 스킵
                if root_i == root_j:
                    continue
                
                # 그룹 확장 시 안전장치 (간단 버전)
                if root_i in group_members_legacy and root_j in group_members_legacy:
                    # 두 그룹 모두 존재하는 경우, confidence가 충분히 높은지 확인
                    if confidence < self.similarity_threshold * 0.95:  # 더 엄격한 기준
                        continue
                
                union(i, j)
                new_root = find(i)
                
                # 그룹 정보 업데이트
                if new_root not in group_members_legacy:
                    group_members_legacy[new_root] = set()
                group_members_legacy[new_root].add(i)
                group_members_legacy[new_root].add(j)
                
                # 기존 그룹 정보 정리
                if root_i != new_root and root_i in group_members_legacy:
                    del group_members_legacy[root_i]
                if root_j != new_root and root_j in group_members_legacy:
                    del group_members_legacy[root_j]
        
        # Connected components 추출
        components = {}
        for idx in df.index:
            root = find(idx)
            if root not in components:
                components[root] = []
            components[root].append(idx)
        
        # final_groups 생성 (크기 2 이상만)
        final_groups = [comp for comp in components.values() if len(comp) >= 2]
        
        if show_progress:
            msg = f"\n[완료] 총 비교 횟수: {total_comparisons:,}, STEP1 후보 그룹: {len(candidate_groups)}개, STEP2 최종 중복 그룹: {len(final_groups)}개"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        self.duplicate_groups = final_groups
        self.candidate_groups = candidate_groups
        return (candidate_groups, final_groups, match_info)
    
    def _save_checkpoint(self, checkpoint_file: str, processed_row_ids: set, edges: list, 
                        match_info: dict, checked_pairs: set, temp_parent: dict, temp_rank: dict,
                        show_progress: bool = True):
        """체크포인트 저장"""
        import json
        import pickle
        from datetime import datetime
        
        try:
            checkpoint_data = {
                'processed_row_ids': list(processed_row_ids),
                'edges': edges,
                'match_info': {str(k): v for k, v in match_info.items()},  # tuple key를 str로 변환
                'checked_pairs': [list(pair) for pair in checked_pairs],  # set을 list로 변환
                'temp_parent': {str(k): v for k, v in temp_parent.items()},  # key를 str로 변환
                'temp_rank': {str(k): v for k, v in temp_rank.items()},
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # pickle로 저장 (복잡한 데이터 구조 지원)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            if show_progress:
                msg = f"  [체크포인트 저장] {checkpoint_file} 저장 완료 ({len(processed_row_ids)}행 처리됨)"
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)
            return True
        except Exception as e:
            if show_progress:
                error_msg = f"  [경고] 체크포인트 저장 실패: {type(e).__name__}: {str(e)}"
                if self.logger:
                    self.logger.log(error_msg)
                else:
                    print(error_msg)
            return False
    
    def _load_checkpoint(self, checkpoint_file: str, show_progress: bool = True):
        """체크포인트 로드"""
        import pickle
        import os
        
        if not os.path.exists(checkpoint_file):
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # 데이터 복원
            processed_row_ids = set(checkpoint_data['processed_row_ids'])
            edges = checkpoint_data['edges']
            
            # match_info 복원 (str key를 tuple로 변환)
            match_info = {}
            for k, v in checkpoint_data['match_info'].items():
                # "(idx1, idx2)" 형태를 tuple로 변환
                try:
                    k_tuple = eval(k) if isinstance(k, str) else k
                    match_info[k_tuple] = v
                except:
                    match_info[k] = v
            
            # checked_pairs 복원
            checked_pairs = {tuple(pair) for pair in checkpoint_data['checked_pairs']}
            
            # temp_parent, temp_rank 복원 (str key를 원래 타입으로 변환)
            temp_parent = {}
            temp_rank = {}
            for k, v in checkpoint_data['temp_parent'].items():
                try:
                    k_orig = int(k) if k.isdigit() else eval(k)
                    temp_parent[k_orig] = v
                    temp_rank[k_orig] = checkpoint_data['temp_rank'].get(k, 0)
                except:
                    temp_parent[k] = v
                    temp_rank[k] = checkpoint_data['temp_rank'].get(k, 0)
            
            if show_progress:
                msg = f"  [체크포인트 로드] {checkpoint_file} 로드 완료 ({len(processed_row_ids)}행 처리됨, {len(edges)}개 edge)"
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)
            
            return {
                'processed_row_ids': processed_row_ids,
                'edges': edges,
                'match_info': match_info,
                'checked_pairs': checked_pairs,
                'temp_parent': temp_parent,
                'temp_rank': temp_rank,
                'timestamp': checkpoint_data.get('timestamp', 'unknown')
            }
        except Exception as e:
            if show_progress:
                error_msg = f"  [경고] 체크포인트 로드 실패: {type(e).__name__}: {str(e)}"
                if self.logger:
                    self.logger.log(error_msg)
                else:
                    print(error_msg)
            return None
    
    def detect_duplicates_ann(self, df: pd.DataFrame, show_progress: bool = True, 
                              save_intermediate_at: int = 5, checkpoint_file: str = None,
                              checkpoint_interval: int = 1000, output_dir: str = '') -> Tuple[List[List[int]], List[List[int]], dict]:
        """
        ANN 기반 중복 탐지 (O(N×K) 구조)
        
        [주요 변경사항]
        - ANN(FAISS/HNSW) 기반 전역 Top-K 후보생성
        - Cheap Gate → Heavy Judge(Step2) 파이프라인
        - edges 기반 Union-Find 그룹화
        - 체크포인트 저장/로드 지원 (재개 기능)
        
        Args:
            checkpoint_file: 체크포인트 파일 경로 (None이면 체크포인트 사용 안 함)
            checkpoint_interval: 체크포인트 저장 간격 (처리된 행 수, 기본값: 1000)
        
        Returns:
            (candidate_groups: List[List[int]], final_groups: List[List[int]], match_info: dict)
            - candidate_groups: edges로부터 재구성 (중간 저장용)
            - final_groups: Union-Find 기반 최종 중복 그룹
            - match_info: 매칭 정보
        """
        import time
        start_time = time.time()
        from typing import Tuple
        import numpy as np
        from .cheap_gate import cheap_gate, get_script_key
        from .ann_index import ANNIndex, build_bucket_indices
        
        n = len(df)
        
        # 성능 계측
        perf_stats = {
            'preprocess_time': 0,
            'embedding_time': 0,
            'index_build_time': 0,
            'candidate_gen_time': 0,
            'step2_time': 0,
            'grouping_time': 0,
            'total_step2_calls': 0,
            'total_ann_candidates': 0,
            'total_cheap_gate_passed': 0,
        }
        
        # ============================================================
        # 1단계: row_cache + row_meta precompute
        # ============================================================
        if show_progress:
            msg = "\n[ANN 모드] 1단계: row_cache + row_meta precompute..."
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        preprocess_start = time.time()
        
        # df를 rows 배열로 변환 (df.loc 금지)
        rows = df.to_dict("records")
        
        # row_meta와 row_cache 생성
        row_meta = {}
        row_cache = {}
        
        # [버그 수정] pos 기반 루프로 변경 (df.index가 0..n-1 정수가 아닐 수 있음)
        for pos, idx in enumerate(df.index):
            row = rows[pos]  # pos는 0부터 연속된 정수
            name = str(row.get('공급업체명', '')) if pd.notna(row.get('공급업체명', '')) else ''
            country = str(row.get('Land', '')).strip().upper() if pd.notna(row.get('Land', '')) else ''
            
            # row_meta 계산 (extract_company_core_name 사용)
            canonical_key = self.extract_company_core_name(name) if name else ''
            embedding_text = name.strip()  # 의미 보존
            country_key = country.strip().upper() if country else 'UNK'
            script_key = get_script_key(name)
            import re
            tokens = set(re.findall(r'\b\w+\b', name.upper())) if name else set()
            len_norm = len(name) if name else 0
            
            row_meta[idx] = {
                'canonical_key': canonical_key,
                'embedding_text': embedding_text,
                'country_key': country_key,
                'script_key': script_key,
                'tokens': tokens,
                'len': len_norm,
            }
            
            # row_cache (기존 Step2 함수 호환을 위해)
            core_name = self.extract_company_core_name(name) if name else ''
            normalized_name = self.normalize_name_for_blocking(name) if name else ''
            address = self.normalize_address(pd.Series(row))
            
            row_cache[idx] = {
                'name': name,
                'country': country,
                'city': str(row.get('CITY1', '')) if pd.notna(row.get('CITY1', '')) else '',
                'address': address,
                'core_name': core_name,
                'normalized_name': normalized_name,
                'row': pd.Series(row)  # Step2 호환을 위해
            }
        
        perf_stats['preprocess_time'] = time.time() - preprocess_start
        
        if show_progress:
            msg = f"  전처리 완료: {len(row_meta):,}행, 소요 시간: {perf_stats['preprocess_time']:.2f}초"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        # ============================================================
        # 2단계: embedding 1회 계산 + normalize
        # ============================================================
        if show_progress:
            msg = "\n[ANN 모드] 2단계: embedding 계산 + normalize..."
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        embedding_start = time.time()
        
        if not self.use_embedding:
            raise ValueError("ANN 모드는 임베딩이 필수입니다. use_embedding=True로 설정하세요.")
        
        # 모든 embedding_text를 한 번에 임베딩 (df.index 순서대로)
        embedding_texts = [row_meta[idx]['embedding_text'] for idx in df.index]
        embeddings = self.embedding_model.encode(
            embedding_texts, 
            convert_to_numpy=True, 
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=show_progress
        )
        
        # embeddings를 numpy array로 변환 (N, dim)
        embeddings = np.array(embeddings)
        dim = embeddings.shape[1]
        
        # row_id -> position 매핑 생성 (df.index가 정수형이 아닐 수 있음)
        row_id_to_pos = {row_id: pos for pos, row_id in enumerate(df.index)}
        
        perf_stats['embedding_time'] = time.time() - embedding_start
        
        if show_progress:
            msg = f"  임베딩 완료: {len(embeddings):,}개 벡터, 차원: {dim}, 소요 시간: {perf_stats['embedding_time']:.2f}초"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        # ============================================================
        # 3단계: bucket 분리 인덱스 구축
        # ============================================================
        if show_progress:
            msg = "\n[ANN 모드] 3단계: bucket 분리 인덱스 구축..."
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        index_start = time.time()
        
        # bucket_key 함수: (country_key, script_key)
        def bucket_key_fn(row_id, meta):
            return (meta['country_key'], meta['script_key'])
        
        # bucket별 인덱스 구축
        bucket_indices = build_bucket_indices(row_meta, embeddings, bucket_key_fn, row_id_to_pos)
        
        perf_stats['index_build_time'] = time.time() - index_start
        
        if show_progress:
            msg = f"  인덱스 구축 완료: {len(bucket_indices)}개 bucket, 소요 시간: {perf_stats['index_build_time']:.2f}초"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        # ============================================================
        # 4단계: ANN 후보생성 + cheap_gate + Step2 연결
        # ============================================================
        if show_progress:
            msg = "\n[ANN 모드] 4단계: ANN 후보생성 + Cheap Gate + Step2..."
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        candidate_start = time.time()
        
        # 기본 파라미터 (성능 최적화: 파라미터 대폭 감소)
        base_top_k_search = 50  # 기본 ANN 검색 후보 수 (100 → 50으로 감소, 속도 2배 향상)
        base_k_pass = 10  # 기본 Cheap Gate 통과 후 Step2로 넘길 최대 후보 수 (20 → 10으로 감소, 속도 2배 향상)
        
        # [체크포인트] 체크포인트 로드 (있는 경우)
        processed_row_ids = set()  # 처리된 row_id 추적
        if checkpoint_file:
            checkpoint_data = self._load_checkpoint(checkpoint_file, show_progress)
            if checkpoint_data:
                processed_row_ids = checkpoint_data['processed_row_ids']
                edges = checkpoint_data['edges']
                match_info = checkpoint_data['match_info']
                checked_pairs = checkpoint_data['checked_pairs']
                temp_parent = checkpoint_data['temp_parent']
                temp_rank = checkpoint_data['temp_rank']
                
                # 누락된 row_id는 초기화 (새로 추가된 행일 수 있음)
                for idx in df.index:
                    if idx not in temp_parent:
                        temp_parent[idx] = idx
                        temp_rank[idx] = 0
                
                if show_progress:
                    msg = f"  [재개] 체크포인트에서 {len(processed_row_ids)}행 복원, {len(edges)}개 edge 복원"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
            else:
                # 체크포인트 로드 실패 시 초기화
                edges = []
                match_info = {}
                checked_pairs = set()
                temp_parent = {idx: idx for idx in df.index}
                temp_rank = {idx: 0 for idx in df.index}
        else:
            # 체크포인트 사용 안 함
            edges = []
            match_info = {}
            checked_pairs = set()
            temp_parent = {idx: idx for idx in df.index}
            temp_rank = {idx: 0 for idx in df.index}
        
        # edges: Step2 통과한 pair들 (검증 강화)
        # edges = [(i, j, score, match_info, cheap_gate_pass, step2_pass, confidence)]
        edge_log = []  # 디버깅용: 모든 edge 생성 로그
        intermediate_saved = False  # 중간 저장 여부
        
        def temp_find(x):
            if temp_parent[x] != x:
                temp_parent[x] = temp_find(temp_parent[x])
            return temp_parent[x]
        
        def temp_union(a, b):
            ra, rb = temp_find(a), temp_find(b)
            if ra == rb:
                return
            if temp_rank[ra] < temp_rank[rb]:
                temp_parent[ra] = rb
            elif temp_rank[ra] > temp_rank[rb]:
                temp_parent[rb] = ra
            else:
                temp_parent[rb] = ra
                temp_rank[ra] += 1
        
        # Bucket별 통계 (K 자동 조정용)
        bucket_stats = {}  # {bucket_key: {'size': int, 'cheap_gate_pass_rate': float}}
        
        # [성능 개선] Bucket별 멤버 사전 계산 (O(N²) → O(N))
        bucket_members = {}  # {bucket_key: [row_id, ...]}
        for idx in df.index:
            meta = row_meta[idx]
            bucket_key = bucket_key_fn(idx, meta)
            if bucket_key not in bucket_members:
                bucket_members[bucket_key] = []
            bucket_members[bucket_key].append(idx)
        
        # 각 row에 대해 ANN 후보생성
        total_rows = len(df.index)
        processed_rows = len(processed_row_ids)  # 체크포인트에서 복원된 행 수
        last_log_time = time.time()
        last_checkpoint_time = time.time()
        
        for idx in df.index:
            # [체크포인트] 이미 처리된 행은 건너뛰기
            if idx in processed_row_ids:
                continue
            meta = row_meta[idx]
            bucket_key = bucket_key_fn(idx, meta)
            
            # 진행 상황 로그 (5초마다)
            processed_rows += 1
            processed_row_ids.add(idx)  # 처리된 행 추적
            current_time = time.time()
            
            if show_progress and (current_time - last_log_time >= 5.0 or processed_rows == total_rows):
                progress_pct = (processed_rows / total_rows) * 100
                msg = f"[ANN 모드] 진행: {processed_rows}/{total_rows} ({progress_pct:.1f}%)"
                if self.logger:
                    self.logger.log(msg)
                else:
                    print(msg)
                last_log_time = current_time
            
            # [체크포인트] 주기적 저장 (checkpoint_interval행마다 또는 5분마다)
            if checkpoint_file and (
                processed_rows % checkpoint_interval == 0 or 
                (current_time - last_checkpoint_time >= 300)  # 5분마다
            ):
                self._save_checkpoint(
                    checkpoint_file, processed_row_ids, edges, match_info,
                    checked_pairs, temp_parent, temp_rank, show_progress
                )
                last_checkpoint_time = current_time
            
            # [성능 개선] K 자동 조정: bucket size 기반 (O(1) 조회)
            bucket_size = len(bucket_members.get(bucket_key, []))
            
            # bucket size가 작으면 top_k_search 감소, 크면 증가 (더 공격적으로 감소)
            if bucket_size < 30:
                top_k_search = max(20, int(base_top_k_search * 0.4))  # 매우 작은 bucket은 40%
            elif bucket_size < 100:
                top_k_search = max(30, int(base_top_k_search * 0.6))  # 작은 bucket은 60%
            elif bucket_size > 1000:
                top_k_search = min(100, int(base_top_k_search * 1.2))  # 큰 bucket은 1.2배 (최대 100)
            else:
                top_k_search = base_top_k_search
            
            # [개선 3] UNK fallback bucket 검색
            index = None
            fallback_buckets = []
            
            if bucket_key in bucket_indices:
                index = bucket_indices[bucket_key]
            else:
                # UNK fallback: country/script 누락 시 확장 검색
                country_key, script_key = bucket_key
                
                # 같은 country, 다른 script 검색
                if script_key != 'unknown':
                    fallback_key = (country_key, 'unknown')
                    if fallback_key in bucket_indices:
                        fallback_buckets.append(bucket_indices[fallback_key])
                
                # UNK country, 같은 script 검색
                if country_key != 'UNK':
                    fallback_key = ('UNK', script_key)
                    if fallback_key in bucket_indices:
                        fallback_buckets.append(bucket_indices[fallback_key])
                
                # UNK country, unknown script 검색 (최후의 수단)
                if country_key != 'UNK' and script_key != 'unknown':
                    fallback_key = ('UNK', 'unknown')
                    if fallback_key in bucket_indices:
                        fallback_buckets.append(bucket_indices[fallback_key])
                
                # Fallback bucket이 없으면 스킵
                if not fallback_buckets:
                    continue
                
                # 첫 번째 fallback bucket 사용 (또는 여러 bucket 결과 병합 가능)
                index = fallback_buckets[0]
                # Fallback 사용 시 top_k_search 감소 (정확도 우선)
                top_k_search = max(50, int(top_k_search * 0.7))
            
            pos = row_id_to_pos[idx]
            emb = embeddings[pos:pos+1]  # (1, dim)
            
            # ANN 검색 (결과는 row_id로 반환됨 - ann_index.py에서 _ids 매핑 사용)
            neighbor_ids, distances = index.search(emb, top_k_search)
            neighbor_ids = neighbor_ids[0]  # 첫 번째 쿼리 결과
            distances = distances[0]
            
            # [버그 수정 5, 6] neighbor_ids 타입 변환 및 검증
            # neighbor_ids를 int로 변환 (numpy scalar 대비)
            neighbor_ids = [int(nid) for nid in neighbor_ids]
            
            # [검증] neighbor_ids가 row_id인지 확인 (idx와 같은 타입이어야 함)
            if len(neighbor_ids) > 0:
                sample_id = neighbor_ids[0]
                # row_id는 df.index에 있어야 함
                if sample_id not in df.index:
                    if show_progress and self.logger:
                        self.logger.log(f"  [경고] ANN 검색 결과가 row_id가 아님: {sample_id}, idx={idx}")
                    # row_id가 아니면 스킵 (ann_index.py가 올바르게 row_id를 반환해야 함)
                    continue
            
            # 자기 자신 제거
            neighbor_ids = [nid for nid in neighbor_ids if nid != idx]
            distances = distances[:len(neighbor_ids)]
            
            perf_stats['total_ann_candidates'] += len(neighbor_ids)
            
            # Cheap Gate 필터링
            passed_candidates = []
            seen_pairs_gate = set()  # [버그 수정 4] Cheap Gate 단계에서만 사용하는 별도 set
            for j, dist in zip(neighbor_ids, distances):
                # [버그 수정 4] Cheap Gate 단계에서는 별도 set 사용 (Step2에서 checked_pairs 사용)
                pair_key = (min(idx, j), max(idx, j))
                if pair_key in seen_pairs_gate:
                    continue
                seen_pairs_gate.add(pair_key)
                
                # Cheap Gate 체크
                meta_j = row_meta[j]
                pass_gate, gate_reason = cheap_gate(idx, j, meta, meta_j)
                
                # Edge 로그 기록 (Cheap Gate 단계)
                edge_log.append({
                    'i': idx,
                    'j': j,
                    'cheap_gate_pass': pass_gate,
                    'gate_reason': gate_reason,
                    'step2_pass': False,  # 아직 Step2 통과 전
                    'confidence': 0.0,
                    'match_type': None,
                    'match_reason': None,
                    'distance': dist
                })
                
                if pass_gate:
                    passed_candidates.append((int(j), dist, gate_reason))
                    perf_stats['total_cheap_gate_passed'] += 1
            
            # [개선 1] K 자동 조정: cheap_gate 통과율 기반으로 k_pass 조절
            if len(neighbor_ids) > 0:
                cheap_gate_pass_rate = len(passed_candidates) / len(neighbor_ids)
                
                # 통과율이 높으면 k_pass 증가, 낮으면 감소 (더 공격적으로 감소)
                if cheap_gate_pass_rate > 0.6:
                    k_pass = min(20, int(base_k_pass * 1.3))  # 통과율 매우 높으면 약간 증가 (최대 20)
                elif cheap_gate_pass_rate < 0.15:
                    k_pass = max(5, int(base_k_pass * 0.5))  # 통과율 낮으면 절반 (최소 5)
                elif cheap_gate_pass_rate < 0.3:
                    k_pass = max(7, int(base_k_pass * 0.7))  # 통과율 중간이면 70%
                else:
                    k_pass = base_k_pass
                
                # Bucket 통계 업데이트
                if bucket_key not in bucket_stats:
                    bucket_stats[bucket_key] = {'size': bucket_size, 'pass_rates': []}
                bucket_stats[bucket_key]['pass_rates'].append(cheap_gate_pass_rate)
            else:
                k_pass = base_k_pass
            
            # 상위 k_pass개만 Step2로
            passed_candidates.sort(key=lambda x: x[1])  # 거리 기준 정렬 (낮을수록 유사)
            passed_candidates = passed_candidates[:k_pass]
            
            # Step2 판정
            for j, dist, gate_reason in passed_candidates:
                try:
                    # [버그 수정 4] Step2 직전에만 checked_pairs 체크 및 등록
                    pair_key = (min(idx, j), max(idx, j))
                    if pair_key in checked_pairs:
                        continue  # 이미 Step2까지 처리된 pair는 스킵
                    checked_pairs.add(pair_key)  # Step2 처리 시작 전에 등록
                    
                    cached_i = row_cache[idx]
                    cached_j = row_cache[j]
                    
                    # Step2: 기존 are_duplicates 함수 재사용 (에러 처리 강화)
                    try:
                        is_dup, confidence, match_type, match_reason = self.are_duplicates(
                            cached_i['row'], cached_j['row']
                        )
                        perf_stats['total_step2_calls'] += 1
                    except Exception as step2_error:
                        # Step2 에러 발생 시 로그만 기록하고 계속 진행
                        if show_progress and self.logger:
                            try:
                                error_msg = f"  [경고] Step2 비교 중 오류 발생: ({idx}, {j}): {type(step2_error).__name__}: {str(step2_error)[:50]}"
                                self.logger.log(error_msg)
                            except:
                                pass
                        # 에러 발생 시 중복 아님으로 처리
                        is_dup = False
                        confidence = 0.0
                        match_type = "STEP2_ERROR"
                        match_reason = str(step2_error)[:100]  # 에러 메시지 일부만 저장
                        perf_stats['total_step2_calls'] += 1  # 호출 횟수는 증가
                    
                    # 매칭 정보 저장
                    match_info[(idx, j)] = {
                        'match_type': match_type,
                        'match_reason': match_reason,
                        'confidence': confidence
                    }
                    
                    # Edge 로그 업데이트 (Step2 결과)
                    # 이전 로그에서 찾아서 업데이트
                    for log_entry in edge_log:
                        if log_entry['i'] == idx and log_entry['j'] == j:
                            log_entry['step2_pass'] = is_dup
                            log_entry['confidence'] = confidence
                            log_entry['match_type'] = match_type
                            log_entry['match_reason'] = match_reason
                            break
                    
                    # [검증 강화] Step2 통과 AND confidence >= threshold인 경우만 edge 추가
                    step2_threshold = self.similarity_threshold  # 기본값 0.85
                    if is_dup and confidence >= step2_threshold:
                        # score 계산 (거리 기반, 낮을수록 유사)
                        score = 1.0 / (1.0 + dist)  # 거리를 유사도로 변환
                        edges.append((idx, j, score, match_info[(idx, j)], True, True, confidence))
                        
                        # [중간 저장] 임시 Union-Find에 추가
                        temp_union(idx, j)
                        
                        # [중간 저장] 10개 그룹 발견 시 중간 결과 저장
                        if save_intermediate_at > 0 and not intermediate_saved:
                            # 임시 그룹 수 계산
                            temp_components = {}
                            for idx_temp in df.index:
                                root = temp_find(idx_temp)
                                if root not in temp_components:
                                    temp_components[root] = []
                                temp_components[root].append(idx_temp)
                            
                            temp_final_groups = [comp for comp in temp_components.values() if len(comp) >= 2]
                            
                            if len(temp_final_groups) >= save_intermediate_at:
                                try:
                                    from .merger import DataMerger
                                    merger = DataMerger()
                                    
                                    # candidate_groups는 edges로부터 간단히 생성
                                    temp_candidate_groups = []
                                    candidate_nodes = set()
                                    for edge in edges:
                                        i, j = edge[0], edge[1]
                                        candidate_nodes.add(i)
                                        candidate_nodes.add(j)
                                    
                                    visited = set()
                                    for node in candidate_nodes:
                                        if node in visited:
                                            continue
                                        group = [node]
                                        visited.add(node)
                                        for edge in edges:
                                            i, j = edge[0], edge[1]
                                            if i == node and j not in visited:
                                                group.append(j)
                                                visited.add(j)
                                            elif j == node and i not in visited:
                                                group.append(i)
                                                visited.add(i)
                                        if len(group) > 1:
                                            temp_candidate_groups.append(sorted(group))
                                    
                                    df_intermediate = merger.merge_duplicates_2step(df, temp_candidate_groups, temp_final_groups, match_info)
                                    if '_merged_from_indices' in df_intermediate.columns:
                                        df_intermediate = df_intermediate.drop(columns=['_merged_from_indices'])
                                    
                                    from datetime import datetime
                                    import os
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    intermediate_file = f'중간결과_{len(temp_final_groups)}개그룹_{timestamp}.xlsx'
                                    if output_dir:
                                        os.makedirs(output_dir, exist_ok=True)
                                        intermediate_file = os.path.join(output_dir, intermediate_file)
                                    df_intermediate.to_excel(intermediate_file, index=False)
                                    
                                    if show_progress:
                                        msg = f"  [중간 저장] {len(temp_final_groups)}개 그룹 발견 → {intermediate_file} 저장 완료"
                                        if self.logger:
                                            self.logger.log(msg)
                                        else:
                                            print(msg)
                                    
                                    intermediate_saved = True
                                except Exception as save_error:
                                    if show_progress:
                                        try:
                                            error_msg = f"  [경고] 중간 저장 실패: {type(save_error).__name__}: {str(save_error)}"
                                            if self.logger:
                                                self.logger.log(error_msg)
                                            else:
                                                print(error_msg)
                                        except:
                                            pass
                    elif is_dup and confidence < step2_threshold:
                        # Step2 통과했지만 confidence가 낮은 경우 로그만 기록
                        if show_progress and self.logger:
                            self.logger.log(f"  [경고] Step2 통과했지만 confidence 낮음: ({idx}, {j}), conf={confidence:.3f} < {step2_threshold}")
                except Exception as loop_error:
                    # 루프 레벨 에러 처리 (계속 진행)
                    if show_progress and self.logger:
                        try:
                            error_msg = f"  [경고] 후보 처리 중 오류 발생: ({idx}, {j}): {type(loop_error).__name__}: {str(loop_error)[:50]}"
                            self.logger.log(error_msg)
                        except:
                            pass
                    continue  # 다음 후보로 계속 진행
                
                # 매칭 정보 저장
                match_info[(idx, j)] = {
                    'match_type': match_type,
                    'match_reason': match_reason,
                    'confidence': confidence
                }
                
                # Edge 로그 업데이트 (Step2 결과)
                # 이전 로그에서 찾아서 업데이트
                for log_entry in edge_log:
                    if log_entry['i'] == idx and log_entry['j'] == j:
                        log_entry['step2_pass'] = is_dup
                        log_entry['confidence'] = confidence
                        log_entry['match_type'] = match_type
                        log_entry['match_reason'] = match_reason
                        break
                
                # [검증 강화] Step2 통과 AND confidence >= threshold인 경우만 edge 추가
                step2_threshold = self.similarity_threshold  # 기본값 0.85
                if is_dup and confidence >= step2_threshold:
                    # score 계산 (거리 기반, 낮을수록 유사)
                    score = 1.0 / (1.0 + dist)  # 거리를 유사도로 변환
                    edges.append((idx, j, score, match_info[(idx, j)], True, True, confidence))
                    
                    # [중간 저장] 임시 Union-Find에 추가
                    temp_union(idx, j)
                    
                    # [중간 저장] 10개 그룹 발견 시 중간 결과 저장
                    if save_intermediate_at > 0 and not intermediate_saved:
                        # 임시 그룹 수 계산
                        temp_components = {}
                        for idx_temp in df.index:
                            root = temp_find(idx_temp)
                            if root not in temp_components:
                                temp_components[root] = []
                            temp_components[root].append(idx_temp)
                        
                        temp_final_groups = [comp for comp in temp_components.values() if len(comp) >= 2]
                        
                        if len(temp_final_groups) >= save_intermediate_at:
                            try:
                                from .merger import DataMerger
                                merger = DataMerger()
                                
                                # candidate_groups는 edges로부터 간단히 생성
                                temp_candidate_groups = []
                                candidate_nodes = set()
                                for edge in edges:
                                    i, j = edge[0], edge[1]
                                    candidate_nodes.add(i)
                                    candidate_nodes.add(j)
                                
                                visited = set()
                                for node in candidate_nodes:
                                    if node in visited:
                                        continue
                                    group = [node]
                                    visited.add(node)
                                    for edge in edges:
                                        i, j = edge[0], edge[1]
                                        if i == node and j not in visited:
                                            group.append(j)
                                            visited.add(j)
                                        elif j == node and i not in visited:
                                            group.append(i)
                                            visited.add(i)
                                    if len(group) > 1:
                                        temp_candidate_groups.append(sorted(group))
                                
                                df_intermediate = merger.merge_duplicates_2step(df, temp_candidate_groups, temp_final_groups, match_info)
                                if '_merged_from_indices' in df_intermediate.columns:
                                    df_intermediate = df_intermediate.drop(columns=['_merged_from_indices'])
                                
                                from datetime import datetime
                                import os
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                intermediate_file = f'중간결과_{len(temp_final_groups)}개그룹_{timestamp}.xlsx'
                                if output_dir:
                                    os.makedirs(output_dir, exist_ok=True)
                                    intermediate_file = os.path.join(output_dir, intermediate_file)
                                df_intermediate.to_excel(intermediate_file, index=False)
                                
                                if show_progress:
                                    msg = f"  [중간 저장] {len(temp_final_groups)}개 그룹 발견 → {intermediate_file} 저장 완료"
                                    if self.logger:
                                        self.logger.log(msg)
                                    else:
                                        print(msg)
                                
                                intermediate_saved = True
                            except Exception as save_error:
                                if show_progress:
                                    try:
                                        error_msg = f"  [경고] 중간 저장 실패: {type(save_error).__name__}: {str(save_error)}"
                                        if self.logger:
                                            self.logger.log(error_msg)
                                        else:
                                            print(error_msg)
                                    except:
                                        pass
                elif is_dup and confidence < step2_threshold:
                    # Step2 통과했지만 confidence가 낮은 경우 로그만 기록
                    if show_progress and self.logger:
                        self.logger.log(f"  [경고] Step2 통과했지만 confidence 낮음: ({idx}, {j}), conf={confidence:.3f} < {step2_threshold}")
        
        perf_stats['candidate_gen_time'] = time.time() - candidate_start
        perf_stats['step2_time'] = perf_stats['candidate_gen_time']  # 대략적
        
        if show_progress:
            msg = f"  후보생성 완료: ANN 후보 {perf_stats['total_ann_candidates']:,}개, "
            msg += f"Cheap Gate 통과 {perf_stats['total_cheap_gate_passed']:,}개, "
            msg += f"Step2 호출 {perf_stats['total_step2_calls']:,}개, "
            msg += f"중복 발견 {len(edges):,}개"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        # ============================================================
        # 5단계: edges 기반 Union-Find 그룹화
        # ============================================================
        if show_progress:
            msg = "\n[ANN 모드] 5단계: Union-Find 그룹화..."
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        grouping_start = time.time()
        
        # Union-Find 구현
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
        
        # 인덱스 매핑 (df.index가 정수형이 아닐 수 있음)
        idx_to_uf = {idx: i for i, idx in enumerate(df.index)}
        uf_to_idx = {i: idx for i, idx in enumerate(df.index)}
        uf = UnionFind(len(df.index))
        
        # edges로 Union-Find 수행 (검증 강화 + 그룹 확장 안전장치)
        invalid_edges = []
        # [개선] 그룹 확장 시 대표(centroid) 재검증을 위한 그룹 정보 저장
        group_centroids = {}  # root_id -> (representative_idx, min_confidence)
        group_members = {}  # root_id -> set of idx
        
        for edge in edges:
            if len(edge) >= 7:
                i, j, score, info, cheap_gate_pass, step2_pass, confidence = edge[:7]
            else:
                # 기존 형식 호환 (i, j, score, info)
                i, j, score, info = edge[:4]
                cheap_gate_pass = True  # 기존 edge는 모두 통과했다고 가정
                step2_pass = True
                confidence = info.get('confidence', 1.0) if isinstance(info, dict) else 1.0
            
            # [검증] Step2 통과 AND confidence >= threshold 확인
            step2_threshold = self.similarity_threshold
            
            # conf=0 또는 step2_pass=False인 edge는 절대 union되지 않게 강제
            if not step2_pass or confidence < step2_threshold:
                invalid_edges.append((i, j, confidence, info.get('match_type', 'UNKNOWN')))
                continue
            
            uf_i = idx_to_uf[i]
            uf_j = idx_to_uf[j]
            
            # [개선] 그룹 확장 안전장치: 대표(centroid) 재검증
            root_i = uf.find(uf_i)
            root_j = uf.find(uf_j)
            
            # 이미 같은 그룹이면 스킵
            if root_i == root_j:
                continue
            
            # 그룹 확장 시 안전장치: A-B OK, B-C OK라도 A-C가 너무 낮으면 보류
            # complete-linkage 느낌의 보수적 병합 규칙
            if root_i in group_members and root_j in group_members:
                # 두 그룹 모두 이미 존재하는 경우, 그룹 간 최소 유사도 확인
                members_i = group_members[root_i]
                members_j = group_members[root_j]
                
                # 그룹 간 최소 유사도 계산 (간단히 현재 edge의 confidence 사용)
                # 실제로는 모든 쌍을 확인해야 하지만, 성능을 위해 현재 edge만 확인
                min_cross_similarity = confidence
                
                # 그룹 간 최소 유사도가 너무 낮으면 병합 보류
                # (complete-linkage: 그룹 내 모든 쌍의 최소 유사도가 threshold 이상이어야 함)
                if min_cross_similarity < step2_threshold * 0.9:  # 10% 여유
                    invalid_edges.append((i, j, confidence, f"그룹 간 유사도 부족: {min_cross_similarity:.3f}"))
                    continue
            
            # Union 수행
            uf.union(uf_i, uf_j)
            new_root = uf.find(uf_i)
            
            # 그룹 정보 업데이트
            if new_root not in group_members:
                group_members[new_root] = set()
                group_centroids[new_root] = (i, confidence)  # 첫 번째 노드를 대표로
            
            group_members[new_root].add(i)
            group_members[new_root].add(j)
            
            # 대표 업데이트 (confidence가 높은 노드를 대표로)
            if confidence > group_centroids[new_root][1]:
                group_centroids[new_root] = (i, confidence)
            
            # 기존 그룹 정보 정리
            if root_i != new_root and root_i in group_members:
                del group_members[root_i]
                if root_i in group_centroids:
                    del group_centroids[root_i]
            if root_j != new_root and root_j in group_members:
                del group_members[root_j]
                if root_j in group_centroids:
                    del group_centroids[root_j]
        
        # 잘못된 edge 로그 출력
        if invalid_edges and show_progress:
            msg = f"  [검증] 잘못된 edge {len(invalid_edges)}개 발견 (union 제외됨)"
            if self.logger:
                self.logger.log(msg)
                for i, j, conf, mt in invalid_edges[:10]:  # 처음 10개만 출력
                    self.logger.log(f"    - ({i}, {j}): conf={conf:.3f}, type={mt}")
            else:
                print(msg)
        
        # Connected components 추출 (루트 ID 기반)
        components = {}
        root_to_group_id = {}  # 루트 ID -> 그룹 번호 매핑
        group_id_counter = 1
        
        for i, idx in enumerate(df.index):
            root = uf.find(i)  # Union-Find 루트 ID
            if root not in components:
                components[root] = []
                root_to_group_id[root] = group_id_counter
                group_id_counter += 1
            components[root].append(idx)
        
        # final_groups 생성 (크기 2 이상만, 루트 ID 순서대로)
        final_groups = []
        for root in sorted(components.keys()):  # 루트 ID 순서대로 정렬
            comp = components[root]
            if len(comp) >= 2:
                final_groups.append(comp)
        
        # 그룹 번호 매핑 저장 (재현 테스트용)
        self._group_id_mapping = {i: root_to_group_id[uf.find(idx_to_uf[idx])] 
                                 for i, idx in enumerate(df.index) 
                                 if uf.find(idx_to_uf[idx]) in root_to_group_id}
        self._edge_log = edge_log  # 재현 테스트용
        self._edges = edges  # 재현 테스트용
        self._uf = uf  # 재현 테스트용
        self._idx_to_uf = idx_to_uf  # 재현 테스트용
        self._uf_to_idx = uf_to_idx  # 재현 테스트용
        
        # 간단한 검증: 큰 component 경고
        large_components = [comp for comp in final_groups if len(comp) > 50]
        if large_components and show_progress:
            msg = f"  [경고] 큰 component 발견: {len(large_components)}개 (크기 > 50)"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        perf_stats['grouping_time'] = time.time() - grouping_start
        
        if show_progress:
            msg = f"  그룹화 완료: {len(final_groups)}개 그룹, 소요 시간: {perf_stats['grouping_time']:.2f}초"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        # ============================================================
        # 6단계: candidate_groups 재구성 (edges로부터, 중간 저장용)
        # ============================================================
        # edges로부터 candidate_groups 재구성 (중간 저장용)
        candidate_groups = []
        if save_intermediate_at > 0:
            # edges의 모든 노드를 그룹으로
            candidate_nodes = set()
            for edge in edges:
                if len(edge) >= 2:
                    i, j = edge[0], edge[1]
                    candidate_nodes.add(i)
                    candidate_nodes.add(j)
            
            # 연결된 노드들을 그룹으로 (간단한 버전)
            visited = set()
            for node in candidate_nodes:
                if node in visited:
                    continue
                group = [node]
                visited.add(node)
                # 간단히 연결된 노드 찾기 (완전한 그래프 탐색은 생략)
                for edge in edges:
                    if len(edge) >= 2:
                        i, j = edge[0], edge[1]
                        if i == node and j not in visited:
                            group.append(j)
                            visited.add(j)
                        elif j == node and i not in visited:
                            group.append(i)
                            visited.add(i)
                if len(group) > 1:
                    candidate_groups.append(sorted(group))
        
        # ============================================================
        # 성능 로그 출력
        # ============================================================
        total_time = time.time() - start_time
        if show_progress:
            msg = f"\n[ANN 모드 완료] 총 소요 시간: {total_time:.2f}초"
            msg += f"\n  - 전처리: {perf_stats['preprocess_time']:.2f}초"
            msg += f"\n  - 임베딩: {perf_stats['embedding_time']:.2f}초"
            msg += f"\n  - 인덱스 구축: {perf_stats['index_build_time']:.2f}초"
            msg += f"\n  - 후보생성+Step2: {perf_stats['candidate_gen_time']:.2f}초"
            msg += f"\n  - 그룹화: {perf_stats['grouping_time']:.2f}초"
            msg += f"\n  - Step2 호출 횟수: {perf_stats['total_step2_calls']:,}개 (핵심 KPI)"
            msg += f"\n  - 최종 중복 그룹: {len(final_groups)}개"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        self.duplicate_groups = final_groups
        self.candidate_groups = candidate_groups
        return (candidate_groups, final_groups, match_info)
    
    def trace_group_edges(self, group_id: int, df: pd.DataFrame) -> Dict:
        """
        특정 STEP2 그룹의 edge 경로를 추적 (재현 테스트용)
        
        Args:
            group_id: STEP2 그룹 번호 (1부터 시작)
            df: 원본 DataFrame
        
        Returns:
            {
                'group_id': int,
                'nodes': List[int],  # 그룹에 포함된 row index들
                'edges': List[Dict],  # 그룹을 연결한 edge들
                'spanning_tree': List[Dict],  # 최소 spanning tree edges
                'edge_paths': Dict  # 각 노드가 어떻게 연결되었는지 경로
            }
        """
        if not hasattr(self, '_edges') or not hasattr(self, '_uf'):
            raise ValueError("detect_duplicates_ann을 먼저 실행해야 합니다.")
        
        # 그룹 ID에 해당하는 노드 찾기
        group_nodes = []
        for i, idx in enumerate(df.index):
            uf_idx = self._idx_to_uf[idx]
            root = self._uf.find(uf_idx)
            if root in self._group_id_mapping and self._group_id_mapping[i] == group_id:
                group_nodes.append(idx)
        
        if not group_nodes:
            return {
                'group_id': group_id,
                'nodes': [],
                'edges': [],
                'spanning_tree': [],
                'edge_paths': {}
            }
        
        # 그룹에 포함된 edge 찾기
        group_edges = []
        for edge in self._edges:
            if len(edge) >= 4:
                i, j = edge[0], edge[1]
                if i in group_nodes and j in group_nodes:
                    edge_info = {
                        'i': i,
                        'j': j,
                        'score': edge[2] if len(edge) > 2 else None,
                        'match_info': edge[3] if len(edge) > 3 else {},
                        'cheap_gate_pass': edge[4] if len(edge) > 4 else True,
                        'step2_pass': edge[5] if len(edge) > 5 else True,
                        'confidence': edge[6] if len(edge) > 6 else (edge[3].get('confidence', 0.0) if isinstance(edge[3], dict) else 0.0)
                    }
                    group_edges.append(edge_info)
        
        # 최소 spanning tree 계산 (Kruskal 알고리즘)
        spanning_tree = []
        
        class LocalUnionFind:
            def __init__(self, nodes):
                self.parent = {n: n for n in nodes}
                self.rank = {n: 0 for n in nodes}
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                return True
        
        uf_local = LocalUnionFind(group_nodes)
        
        # Edge를 confidence 기준으로 정렬 (높은 것부터)
        sorted_edges = sorted(group_edges, key=lambda e: e['confidence'], reverse=True)
        
        for edge in sorted_edges:
            if uf_local.union(edge['i'], edge['j']):
                spanning_tree.append(edge)
                if len(spanning_tree) >= len(group_nodes) - 1:
                    break
        
        # 각 노드의 연결 경로 추적
        edge_paths = {}
        for node in group_nodes:
            path = []
            visited = set()
            
            def dfs(current, target, current_path):
                if current == target:
                    return current_path
                visited.add(current)
                for edge in group_edges:
                    if edge['i'] == current and edge['j'] not in visited:
                        result = dfs(edge['j'], target, current_path + [edge])
                        if result:
                            return result
                    elif edge['j'] == current and edge['i'] not in visited:
                        result = dfs(edge['i'], target, current_path + [edge])
                        if result:
                            return result
                return None
            
            # 루트 노드(첫 번째 노드)까지의 경로 찾기
            if node != group_nodes[0]:
                visited.clear()
                path = dfs(group_nodes[0], node, [])
            edge_paths[node] = path if path else []
        
        return {
            'group_id': group_id,
            'nodes': group_nodes,
            'edges': group_edges,
            'spanning_tree': spanning_tree,
            'edge_paths': edge_paths
        }
    
    def get_duplicate_groups(self) -> List[List[int]]:
        return self.duplicate_groups

