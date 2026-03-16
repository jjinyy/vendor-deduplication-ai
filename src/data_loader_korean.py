"""
한국 업체 전용 데이터 로더
- 입력 컬럼: 공급업체코, 공급업체명, 주소 (한국 형식)
- 기존 하이브리드 탐지기가 기대하는 스키마로 변환하여 반환
"""
import pandas as pd
import re
import os
from typing import Optional


# 기존 detector가 참조하는 주소/메타 필드 (없으면 NaN으로 채움)
DETECTOR_ADDRESS_FIELDS = [
    'CITY1', 'CITY2', 'STREET', 'HOUSE_NUM1', 'HOUSE_NUM2',
    'STR_SUPPL1', 'STR_SUPPL2', 'STR_SUPPL3',
]


def parse_korean_city_from_address(address: str) -> str:
    """
    한국 주소에서 시/군/구 단위 추출 (CITY1 용).
    예: "경기도 부천시 소사구 양지로 237" -> "부천시 소사구"
    """
    if not address or not isinstance(address, str):
        return ""
    s = address.strip()
    if not s:
        return ""
    # 시/군/구 패턴: 한글+시, 한글+군, 한글+구
    parts = []
    for m in re.finditer(r'([가-힣]+(?:시|군|구))', s):
        parts.append(m.group(1))
    return ' '.join(parts[:2]) if parts else ""  # 최대 2개 (시, 구 등)


class DataLoaderKorean:
    """
    한국 업체 파일 전용 로더.
    컬럼: 공급업체코, 공급업체명, 주소 -> 기존 하이브리드 탐지기 스키마로 매핑.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """파일 로드 후 공통 스키마 DataFrame 반환"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")

        file_ext = os.path.splitext(self.file_path)[1].lower()

        if file_ext == '.csv':
            try:
                raw = pd.read_csv(self.file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                try:
                    raw = pd.read_csv(self.file_path, encoding='cp949')
                except UnicodeDecodeError:
                    raw = pd.read_csv(self.file_path, encoding='utf-8')
        elif file_ext in ['.xlsx', '.xls']:
            raw = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")

        raw = raw.replace(['', ' ', '  '], None)

        # 컬럼명 정규화 (공백/오타 허용)
        col_map = {}
        for c in raw.columns:
            c_clean = str(c).strip()
            if c_clean in ('공급업체코', '공급업체코드'):
                col_map[c] = '공급업체코'
            elif c_clean == '공급업체명':
                col_map[c] = '공급업체명'
            elif c_clean == '주소':
                col_map[c] = '주소'
        raw = raw.rename(columns=col_map)

        # 필수 컬럼 확인
        if '공급업체명' not in raw.columns:
            raise ValueError("필수 컬럼이 없습니다: 공급업체명")
        if '주소' not in raw.columns:
            raise ValueError("필수 컬럼이 없습니다: 주소")

        # 코드 컬럼: 없으면 인덱스로 대체
        if '공급업체코' not in raw.columns:
            raw['공급업체코'] = raw.index.astype(str)

        # 공통 스키마로 변환 (기존 duplicate_detector_hybrid / merger 호환)
        rows = []
        for i, r in raw.iterrows():
            code = r.get('공급업체코')
            if pd.isna(code):
                code = i
            name = r.get('공급업체명')
            if pd.isna(name):
                name = ''
            else:
                name = str(name).strip()
            addr = r.get('주소')
            if pd.isna(addr):
                addr = ''
            else:
                addr = str(addr).strip()

            city1 = parse_korean_city_from_address(addr)

            row = {
                '공급업체코드': code,   # merger/detector에서 사용
                '공급업체명': name,
                'Land': 'KR',
                'CITY1': city1,
                'STR_SUPPL1': addr,    # 전체 주소 (normalize_address에서 사용)
            }
            for f in DETECTOR_ADDRESS_FIELDS:
                if f not in row:
                    row[f] = None
            row['CITY2'] = None
            row['STREET'] = None
            row['HOUSE_NUM1'] = None
            row['HOUSE_NUM2'] = None
            row['STR_SUPPL2'] = None
            row['STR_SUPPL3'] = None
            rows.append(row)

        self.df = pd.DataFrame(rows)
        # 인덱스는 기본 0..n-1 유지 (공급업체코드는 컬럼으로만 사용, 중복 코드 시 인덱스 충돌 방지)
        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load()를 먼저 호출하세요.")
        return self.df
