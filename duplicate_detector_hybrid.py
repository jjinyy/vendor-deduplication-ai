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
        """포르투갈어/베트남어 악센트 제거"""
        if not text:
            return ""
        
        # 포르투갈어 악센트 매핑
        portuguese_accents = {
            'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o', 'ö': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n',
            'Á': 'A', 'À': 'A', 'Ã': 'A', 'Â': 'A', 'Ä': 'A',
            'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
            'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
            'Ó': 'O', 'Ò': 'O', 'Õ': 'O', 'Ô': 'O', 'Ö': 'O',
            'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
            'Ç': 'C', 'Ñ': 'N'
        }
        
        # 베트남어 악센트 매핑
        vietnamese_accents = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd',
            'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
            'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
            'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
            'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
            'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
            'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
            'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
            'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
            'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
            'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
            'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
            'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
            'Đ': 'D'
        }
        
        # 모든 악센트 제거
        all_accents = {**portuguese_accents, **vietnamese_accents}
        for accented, unaccented in all_accents.items():
            text = text.replace(accented, unaccented)
        
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
        """다중 Blocking 키 생성 (도시명 로마자화 포함)"""
        keys = []
        country = ""
        if 'Land' in row.index and pd.notna(row['Land']):
            country = str(row['Land']).strip().upper()
        
        # 도시명 로마자화 버전들 추출
        city_romanized_list = self.extract_city_romanized(row)
        # 원본 도시명도 추가
        city_original = self.extract_city(row)
        if city_original:
            city_romanized_list.append(city_original)
        city_romanized_list = list(set(city_romanized_list))  # 중복 제거
        
        name = row.get('공급업체명', '')
        
        if pd.notna(name) and name:
            name_prefixes = self.extract_multilingual_prefixes(str(name), length=5)
            if name_prefixes:
                for prefix in name_prefixes:
                    if prefix:
                        # 도시명이 있으면 각 로마자화 버전과 조합
                        if city_romanized_list:
                            for city_roman in city_romanized_list:
                                keys.append(f"{country}|{prefix}|{city_roman}")
                        else:
                            # 도시명이 없으면 prefix만 사용
                            keys.append(f"{country}|{prefix}|")
            else:
                # prefix가 없으면 도시명만 사용하지 않음 (같은 도시의 다른 업체 오매칭 방지)
                # 대신 국가만 사용하거나 키를 생성하지 않음
                # 이름이 있지만 prefix 추출 실패한 경우는 국가만 사용
                if country:
                    keys.append(f"{country}||")
                # 도시명만으로는 키를 생성하지 않음 (오매칭 방지)
        else:
            # 이름이 없으면 도시명만 사용하지 않음 (오매칭 방지)
            # 국가만 사용하거나 키를 생성하지 않음
            if country:
                keys.append(f"{country}||")
            # 도시명만으로는 키를 생성하지 않음
        
        # 다국어 비교를 위해 국가별 그룹도 추가 (같은 국가 내에서 모든 업체 비교 가능)
        # 하지만 이렇게 하면 너무 큰 그룹이 생성되므로, 이름 prefix가 있을 때만 추가
        # 이렇게 하면 "삼성전자"와 "Samsung Electronics"가 같은 국가(KR)에 있고 prefix가 비슷하면 비교됨
        if country and name_prefixes:
            # prefix가 있을 때만 국가 전체 키 추가 (너무 큰 그룹 방지)
            # 하지만 이렇게 해도 큰 그룹이 생성될 수 있으므로, 큰 그룹 최적화가 필요함
            # keys.append(f"{country}|*|*")  # 주석 처리: 너무 큰 그룹 생성 방지
            pass
        
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
        
        import re
        
        name_clean = name.strip()
        
        # 1. 지역명 제거 (중국어)
        chinese_regions = ['山东', '北京', '上海', '聊城', '沈阳', '广州', '深圳', '杭州', '成都', 
                          '武汉', '西安', '南京', '天津', '重庆', '苏州', '青岛', '大连', '宁波',
                          '厦门', '福州', '济南', '郑州', '长沙', '石家庄', '哈尔滨', '长春',
                          '太原', '合肥', '南昌', '昆明', '贵阳', '南宁', '海口', '兰州', '西宁',
                          '银川', '乌鲁木齐', '拉萨', '呼和浩特']
        
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
        indonesian_legal_forms = ['PT', 'PT.', 'CV', 'CV.', 'TB', 'TB.', 'UD', 'UD.']
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
        # 인도네시아 일반 단어 (고유 상호가 아님 - GPT 피드백 반영)
        # SEJAHTERA, MAKMUR, ABADI, JAYA, SENTOSA, MANDIRI는 수천 개 회사가 사용하는 일반 단어
        # 이 단어들은 "OO상사", "OO산업" 급의 일반 수식어로 고유 상호가 아님
        indonesian_common_words = ['SEJAHTERA', 'MAKMUR', 'ABADI', 'JAYA', 'SENTOSA', 'MANDIRI']
        
        # 모든 불용어 통합 (긴 단어부터 제거하기 위해 정렬)
        all_stopwords = (chinese_regions + chinese_legal_forms + chinese_business_types +
                        english_legal_forms + english_business_types +
                        portuguese_legal_forms + portuguese_business_types +
                        vietnamese_legal_forms + korean_legal_forms +
                        indonesian_legal_forms + thai_legal_forms +
                        indonesian_common_words)
        
        # 악센트 제거 후 정규화
        name_clean_normalized = self.remove_accents(name_clean.upper())
        
        # 불용어 제거 (순서 중요: 긴 단어부터 제거)
        all_stopwords_sorted = sorted(all_stopwords, key=len, reverse=True)
        
        for stopword in all_stopwords_sorted:
            # 원본에서 제거
            name_clean = name_clean.replace(stopword, ' ')
            name_clean = name_clean.replace(stopword.upper(), ' ')
            name_clean = name_clean.replace(stopword.lower(), ' ')
            # 정규화된 버전에서도 제거
            stopword_normalized = self.remove_accents(stopword.upper())
            if stopword_normalized:
                name_clean_normalized = name_clean_normalized.replace(stopword_normalized, ' ')
        
        # 특별 처리: "CO"는 맨 끝에 있을 때만 제거
        # 예: "ABC CO" → "ABC", "ABC CO.LTD" → "ABC"
        # 하지만 "IND.ENG.CO" 같은 경우는 제거하지 않음
        import re
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
        
        return name_clean
    
    def check_core_name_match(self, name1: str, name2: str) -> Tuple[bool, float]:
        """
        고유 상호 비교 (GPT 프롬프트 기반)
        - 핵심 단어가 명확히 다르면 다른 업체
        - 철자 차이, 단어 순서 차이, 구두점 차이는 허용
        - 완전히 다른 단어 조합은 허용하지 않음
        - 다국어 처리: 같은 업체의 다른 언어 표기도 탐지
        """
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
            # 예: "DAESUNG MACHINERY" vs "DAESUNG TECHNOLOGY E&C"
            # 공통 브랜드명 찾기 (대문자로만 구성된 4자 이상 단어)
            common_brands = [w for w in common_words if len(w) >= 4 and w.isupper()]
            if common_brands:
                # 공통 브랜드명이 있으면 기준 완화: 0.65 → 0.60
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
            # 일반 단어 블랙리스트 (고유 상호가 아닌 일반 단어)
            general_words = {
                'INDONESIA', 'INDONESIAN', 'INDONESIA', 'ID',
                'CHINA', 'CHINESE', 'CN',
                'KOREA', 'KOREAN', 'KR',
                'JAPAN', 'JAPANESE', 'JP',
                'THAILAND', 'THAI', 'TH',
                'VIETNAM', 'VIETNAMESE', 'VN',
                'SOLUSI', 'SOLUTION', 'SOLUTIONS', 'SOLUSI',
                'INTEGRASI', 'INTEGRATION', 'INTEGRASI',
                'TEKNOLOGI', 'TECHNOLOGY', 'TECH',
                'PART', 'PARTS',
                'CENTRAL', 'CENTRE', 'CENTER',  # 일반 단어일 수 있음
                'INTERNATIONAL', 'GLOBAL', 'WORLD',
                'GROUP', 'HOLDINGS', 'ENTERPRISES',
            }
            
            # 공통 단어에서 일반 단어 제외
            unique_common_words = common_words - general_words
            
            # 고유 브랜드명만 공통으로 있으면 허용
            if unique_common_words:
                common_word_list = list(unique_common_words)
                # 공통 단어가 브랜드명일 가능성이 높으면 (4자 이상)
                if any(len(w) >= 4 for w in common_word_list):
                    if core_similarity >= 0.65:
                        return (True, core_similarity)
            # 일반 단어만 공통이면 제외
            elif common_words.issubset(general_words):
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
    
    def is_candidate(self, row1: pd.Series, row2: pd.Series) -> Tuple[bool, str]:
        """
        STEP 1: 중복 후보(CANDIDATE) 여부 판단 (Recall 중심, 하지만 엄격한 기준)
        GPT 프롬프트 기반: 같은 업체일 가능성이 있어 STEP2로 넘길 가치가 있는지만 판단
        
        중요 원칙:
        - 공통된 일반 단어, 지역명, 법인 형태 때문에 후보로 판단하지 않음
        - 특히 인도네시아, 중국, 베트남 데이터에서 일반 단어로 인한 오탐 방지
        
        Returns:
            (is_candidate: bool, reason: str)
        """
        name1 = row1.get('공급업체명', '')
        name2 = row2.get('공급업체명', '')
        
        # STEP1 전용 STOP WORD 리스트 (더 엄격)
        step1_stop_words = {
            # 국가명, 지역명
            'INDONESIA', 'INDONESIAN', 'ID', 'CHINA', 'CHINESE', 'CN',
            'KOREA', 'KOREAN', 'KR', 'JAPAN', 'JAPANESE', 'JP',
            'THAILAND', 'THAI', 'TH', 'VIETNAM', 'VIETNAMESE', 'VN',
            # 인도네시아 STOP WORD
            'CENTRAL', 'CENTRA', 'SEJAHTERA', 'MAKMUR', 'ABADI', 'JAYA',
            'MANDIRI', 'INDUSTRIAL', 'PART', 'INTERNATIONAL', 'NIAGA',
            # 중국 STOP WORD (로마자화된 형태)
            'SHANGMAO', 'WUZI', 'JITUAN', 'ZHIZAOCHANG', 'JIANSHEGONGCHENG',
            # 베트남 STOP WORD
            'CTY', 'CONG', 'TY', 'TNHH', 'SX', 'TM', 'DV', 'BAO', 'BI',
            # 영어/포르투갈어 STOP WORD
            'TRADING', 'COMMERCE', 'INDUSTRIAL', 'SOLUTION', 'SOLUTIONS', 'INTEGRATION',
            'SOLUSI', 'INTEGRASI', 'TEKNOLOGI', 'TECHNOLOGY', 'TECH', 'PARTS',
            'CENTRE', 'CENTER', 'GLOBAL', 'WORLD', 'GROUP', 'HOLDINGS', 'ENTERPRISES',
        }
        
        # 1. 업체명 전처리: 고유 상호 추출 (법인 형태, 지역명, 일반 단어 제거)
        core1 = self.extract_company_core_name(str(name1) if pd.notna(name1) else '')
        core2 = self.extract_company_core_name(str(name2) if pd.notna(name2) else '')
        
        # 고유 상호가 없으면 STEP1 후보에서 제외
        if not core1 or not core2:
            return (False, "고유 상호 추출 실패")
        
        import re
        # 고유 상호를 토큰으로 분리
        words1 = set(re.findall(r'\b\w+\b', core1.upper()))
        words2 = set(re.findall(r'\b\w+\b', core2.upper()))
        
        # STOP WORD 제거
        unique_words1 = words1 - step1_stop_words
        unique_words2 = words2 - step1_stop_words
        
        # 고유 단어가 하나도 없으면 제외
        if not unique_words1 or not unique_words2:
            return (False, "고유 단어 없음 (STOP WORD만 존재)")
        
        # 조건 (A): 고유 상호 토큰이 1개 이상 의미 있게 겹침 (STOP WORD 제외)
        unique_common_words = unique_words1 & unique_words2
        if unique_common_words:
            return (True, f"고유 단어 공통: {unique_common_words}")
        
        # 조건 (B): 고유 상호가 축약/확장 관계일 가능성 확인
        # 예: TRIDUTA ↔ TRI DUTA, LONG-LITE ↔ LONGLITE
        core1_no_space = re.sub(r'[\s\-]', '', core1.upper())
        core2_no_space = re.sub(r'[\s\-]', '', core2.upper())
        
        # 한쪽이 다른 쪽의 부분 문자열인지 확인 (최소 4글자 이상)
        if len(core1_no_space) >= 4 and len(core2_no_space) >= 4:
            if core1_no_space in core2_no_space or core2_no_space in core1_no_space:
                # STOP WORD가 포함되어 있지 않은지 확인
                if not any(stop in core1_no_space for stop in step1_stop_words) and \
                   not any(stop in core2_no_space for stop in step1_stop_words):
                    return (True, f"축약/확장 관계 가능성: {core1_no_space} ↔ {core2_no_space}")
        
        # 철자 유사도 확인 (축약/확장 관계)
        from rapidfuzz import fuzz
        if len(core1_no_space) >= 4 and len(core2_no_space) >= 4:
            ratio = fuzz.ratio(core1_no_space, core2_no_space) / 100.0
            # 높은 유사도이면서 STOP WORD가 포함되지 않은 경우
            if ratio >= 0.75:
                # STOP WORD가 포함되어 있지 않은지 확인
                if not any(stop in core1_no_space for stop in step1_stop_words) and \
                   not any(stop in core2_no_space for stop in step1_stop_words):
                    return (True, f"고유 상호 유사도 높음: {ratio:.3f}")
        
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
                            non_stop_common = all_common_words - step1_stop_words
                            # STOP WORD가 아닌 공통 단어가 있으면 후보로 판단
                            if non_stop_common:
                                return (True, f"같은 국가({land1}) 및 도시({city1}), 이름 유사도: {name_sim:.3f}, 공통 단어: {non_stop_common}")
                        else:
                            # 공통 단어가 없어도 이름 유사도가 매우 높으면 (0.85 이상) 후보로 판단
                            # 예: 축약/확장 관계이지만 단어 분리로 인해 공통 단어가 없는 경우
                            if name_sim >= 0.85:
                                return (True, f"같은 국가({land1}) 및 도시({city1}), 이름 유사도 매우 높음: {name_sim:.3f}")
        
        return (False, "후보 조건 불만족")
    
    def are_duplicates(self, row1: pd.Series, row2: pd.Series) -> Tuple[bool, float]:
        """
        STEP 2: 동일 법인(SAME_ENTITY) 최종 판단 (Precision 중심)
        GPT 프롬프트 기반: 실제로 병합해도 되는 동일 법인인지 판단
        
        핵심 원칙:
        1. 고유 상호가 일치해야 함 (필수)
        2. 주소는 보조 지표 (주소만 같고 업체명이 다르면 동일 업체로 판단하지 않음)
        3. 업종 충돌 확인
        4. 확실하지 않으면 같은 업체로 판단하지 않음 (False Positive 방지)
        """
        name1 = row1.get('공급업체명', '')
        name2 = row2.get('공급업체명', '')
        
        # 1단계: 고유 상호 추출 및 비교
        core_name_match, core_similarity = self.check_core_name_match(
            str(name1) if pd.notna(name1) else '',
            str(name2) if pd.notna(name2) else ''
        )
        
        # 고유 상호가 일치하지 않으면 다른 업체 (필수 조건)
        # GPT 피드백: 고유 상호가 다르면 절대 같은 업체로 판단하지 않음
        if not core_name_match:
            return (False, 0.0)
        
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
            
            # 일반 단어 블랙리스트 (고유 상호가 아닌 일반 단어)
            general_words = {
                'INDONESIA', 'INDONESIAN', 'ID',
                'CHINA', 'CHINESE', 'CN',
                'KOREA', 'KOREAN', 'KR',
                'JAPAN', 'JAPANESE', 'JP',
                'THAILAND', 'THAI', 'TH',
                'VIETNAM', 'VIETNAMESE', 'VN',
                'SOLUSI', 'SOLUTION', 'SOLUTIONS',
                'INTEGRASI', 'INTEGRATION',
                'TEKNOLOGI', 'TECHNOLOGY', 'TECH',
                'PART', 'PARTS',
                'CENTRAL', 'CENTRE', 'CENTER',  # 일반 단어일 수 있음
                'INTERNATIONAL', 'GLOBAL', 'WORLD',
                'GROUP', 'HOLDINGS', 'ENTERPRISES',
            }
            
            # 일반 단어 제외한 고유 단어만 비교
            unique_words1 = words1 - general_words
            unique_words2 = words2 - general_words
            unique_common_words = unique_words1 & unique_words2
            all_unique_words = unique_words1 | unique_words2
            
            if all_unique_words:
                unique_common_ratio = len(unique_common_words) / len(all_unique_words)
                # 고유 단어 공통 비율이 30% 미만이면 고유 상호가 완전히 다름
                # 그룹 2, 3, 5 케이스: 일반 단어만 공통 → 다른 업체
                if unique_common_ratio < 0.30:
                    return (False, 0.0)
            elif not unique_words1 or not unique_words2:
                # 고유 단어가 하나도 없으면 (일반 단어만 있으면) 다른 업체
                return (False, 0.0)
        
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
                    return (False, 0.0)
        
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
                    return (False, 0.0)
        
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
        
        # 최종 판단 기준 (GPT 프롬프트 기반, 완화된 버전)
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
        
        # 최종 판단: 고유 상호 일치 + 주소/업종 일관성
        if core_name_match and address_consistent:
            # 신뢰도 계산
            confidence = core_similarity * 0.5
            if code_match:
                confidence += 0.3
            elif building_match:
                confidence += building_score * 0.2
            elif address_similarity >= 0.90:
                confidence += address_similarity * 0.2
            elif address_similarity >= 0.75:
                confidence += address_similarity * 0.15
            else:
                confidence += address_similarity * 0.1
            
            # 이름 유사도 보너스
            if name_similarity >= 0.90:
                confidence += 0.1
            elif name_similarity >= 0.85:
                confidence += 0.05
            
            return (True, min(confidence, 1.0))
        
        # 확실하지 않으면 같은 업체로 판단하지 않음
        return (False, 0.0)
    
    def detect_duplicates(self, df: pd.DataFrame, show_progress: bool = True, 
                         save_intermediate_at: int = 5) -> Tuple[List[List[int]], List[List[int]]]:
        """
        2단계 중복 탐지 (GPT 프롬프트 기반)
        
        Returns:
            (candidate_groups: List[List[int]], final_groups: List[List[int]])
            - candidate_groups: STEP1 후보 그룹 리스트
            - final_groups: STEP2 최종 중복 그룹 리스트
        """
        """
        하이브리드 중복 탐지 (2단계 구조)
        
        Args:
            df: 입력 DataFrame
            show_progress: 진행 상황 표시 여부
            save_intermediate_at: 중간 결과 저장 시점 (중복 그룹 개수)
        """
        from typing import Tuple
        n = len(df)
        candidate_visited = set()  # STEP1 후보 방문 기록
        final_visited = set()  # STEP2 최종 방문 기록
        candidate_groups = []  # STEP1 후보 그룹
        final_groups = []  # STEP2 최종 중복 그룹
        intermediate_saved = False  # 중간 저장 여부
        
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
        large_group_threshold = 50  # 큰 그룹 임계값 (더 일찍 최적화 모드 진입)
        
        for blocking_key, indices in blocking_groups.items():
            block_num += 1
            block_size = len(indices)
            
            if block_size < 2:
                continue
            
            comparisons_in_block = block_size * (block_size - 1) // 2
            total_comparisons += comparisons_in_block
            
            # 진행 상황 로그 (더 자주 기록)
            if show_progress:
                if block_num % 100 == 0 or block_size > large_group_threshold:
                    msg = f"  처리 중: {block_num}/{total_blocks} 그룹 (현재 그룹 크기: {block_size}행), 총 비교 횟수: {total_comparisons:,}"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
            
            # 큰 그룹에 대한 최적화: 샘플링 또는 배치 처리
            if block_size > large_group_threshold:
                if show_progress:
                    msg = f"  [큰 그룹 처리] {block_size}행 그룹 처리 중... (최적화 모드)"
                    if self.logger:
                        self.logger.log(msg)
                    else:
                        print(msg)
                
                # 큰 그룹은 더 효율적으로 처리: 먼저 빠른 비교로 후보 추출
                indices_list = list(indices)
                processed_in_block = set()
                
                for idx, i in enumerate(indices_list):
                    if i in candidate_visited:
                        continue
                    
                    current_candidate_group = [i]
                    candidate_visited.add(i)
                    processed_in_block.add(i)
                    
                    # 큰 그룹에서는 부분 비교만 수행 (처음 50개와 비교, 더 적은 비교로 속도 향상)
                    compare_targets = [j for j in indices_list if j > i and j not in candidate_visited][:50]
                    
                    for j in compare_targets:
                        try:
                            # STEP 1: 후보 여부 판단
                            is_candidate, reason = self.is_candidate(df.loc[i], df.loc[j])
                            
                            if is_candidate:
                                if j not in current_candidate_group:
                                    current_candidate_group.append(j)
                                candidate_visited.add(j)
                                processed_in_block.add(j)
                                
                                # STEP 2: 후보인 경우에만 최종 판단
                                is_dup, confidence = self.are_duplicates(df.loc[i], df.loc[j])
                                
                                if is_dup:
                                    # 최종 중복 그룹에 추가
                                    if i not in final_visited:
                                        final_group = [i]
                                        final_visited.add(i)
                                        final_groups.append(final_group)
                                    else:
                                        # 이미 그룹에 있으면 해당 그룹 찾기
                                        final_group = None
                                        for fg in final_groups:
                                            if i in fg:
                                                final_group = fg
                                                break
                                    
                                    if final_group and j not in final_group:
                                        final_group.append(j)
                                        final_visited.add(j)
                                    
                                    if show_progress and len(final_group) == 2:
                                        try:
                                            name1 = str(df.loc[i].get('공급업체명', ''))[:40]
                                            name2 = str(df.loc[j].get('공급업체명', ''))[:40]
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
                    
                    # STEP1 후보 그룹 추가
                    if len(current_candidate_group) > 1:
                        candidate_groups.append(current_candidate_group)
                        
                        # 중간 결과 저장 (5개 그룹 발견 시)
                        if save_intermediate_at > 0 and len(final_groups) == save_intermediate_at and not intermediate_saved:
                            try:
                                from merger import DataMerger
                                merger = DataMerger()
                                df_intermediate = merger.merge_duplicates_2step(df, candidate_groups, final_groups)
                                if '_merged_from_indices' in df_intermediate.columns:
                                    df_intermediate = df_intermediate.drop(columns=['_merged_from_indices'])
                                
                                from datetime import datetime
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                intermediate_file = f'중간결과_{len(final_groups)}개그룹_{timestamp}.xlsx'
                                df_intermediate.to_excel(intermediate_file, index=False)
                                
                                if show_progress:
                                    msg = f"  [중간 저장] {len(final_groups)}개 그룹 발견 → {intermediate_file} 저장 완료"
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
                    
                    # 진행 상황 업데이트 (큰 그룹에서는 더 자주)
                    if (idx + 1) % 50 == 0 and show_progress:
                        msg = f"    큰 그룹 진행: {idx + 1}/{block_size}행 처리 완료"
                        if self.logger:
                            self.logger.log(msg)
                        else:
                            print(msg)
            else:
                # 작은 그룹은 기존 방식대로 처리 (2단계 구조)
                for i in indices:
                    if i in candidate_visited:
                        continue
                    
                    current_candidate_group = [i]
                    candidate_visited.add(i)
                    
                    for j in indices:
                        if j <= i or j in candidate_visited:
                            continue
                        
                        try:
                            # STEP 1: 후보 여부 판단
                            is_candidate, reason = self.is_candidate(df.loc[i], df.loc[j])
                            
                            if is_candidate:
                                if j not in current_candidate_group:
                                    current_candidate_group.append(j)
                                candidate_visited.add(j)
                                
                                # STEP 2: 후보인 경우에만 최종 판단
                                is_dup, confidence = self.are_duplicates(df.loc[i], df.loc[j])
                                
                                if is_dup:
                                    # 최종 중복 그룹에 추가
                                    if i not in final_visited:
                                        final_group = [i]
                                        final_visited.add(i)
                                        final_groups.append(final_group)
                                    else:
                                        # 이미 그룹에 있으면 해당 그룹 찾기
                                        final_group = None
                                        for fg in final_groups:
                                            if i in fg:
                                                final_group = fg
                                                break
                                    
                                    if final_group and j not in final_group:
                                        final_group.append(j)
                                        final_visited.add(j)
                                    
                                    if show_progress and len(final_group) == 2:
                                        try:
                                            name1 = str(df.loc[i].get('공급업체명', ''))[:40]
                                            name2 = str(df.loc[j].get('공급업체명', ''))[:40]
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
                    
                    # STEP1 후보 그룹 추가
                    if len(current_candidate_group) > 1:
                        candidate_groups.append(current_candidate_group)
                        
                        # 중간 결과 저장 (5개 그룹 발견 시)
                        if save_intermediate_at > 0 and len(final_groups) == save_intermediate_at and not intermediate_saved:
                            try:
                                from merger import DataMerger
                                merger = DataMerger()
                                df_intermediate = merger.merge_duplicates_2step(df, candidate_groups, final_groups)
                                if '_merged_from_indices' in df_intermediate.columns:
                                    df_intermediate = df_intermediate.drop(columns=['_merged_from_indices'])
                                
                                from datetime import datetime
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                intermediate_file = f'중간결과_{len(final_groups)}개그룹_{timestamp}.xlsx'
                                df_intermediate.to_excel(intermediate_file, index=False)
                                
                                if show_progress:
                                    msg = f"  [중간 저장] {len(final_groups)}개 그룹 발견 → {intermediate_file} 저장 완료"
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
        
        if show_progress:
            msg = f"\n[완료] 총 비교 횟수: {total_comparisons:,}, STEP1 후보 그룹: {len(candidate_groups)}개, STEP2 최종 중복 그룹: {len(final_groups)}개"
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)
        
        self.duplicate_groups = final_groups
        self.candidate_groups = candidate_groups
        return (candidate_groups, final_groups)
    
    def get_duplicate_groups(self) -> List[List[int]]:
        return self.duplicate_groups

