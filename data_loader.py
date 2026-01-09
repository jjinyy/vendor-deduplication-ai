"""
데이터 로더 모듈
CSV 및 Excel 파일을 로드하고 전처리합니다.
"""
import pandas as pd
from typing import Optional
import os


class DataLoader:
    """협력사 데이터를 로드하는 클래스"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: 데이터 파일 경로 (CSV 또는 Excel)
        """
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """파일을 로드하고 DataFrame으로 반환"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
        
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.csv':
            # CSV 파일 로드 (인코딩 자동 감지 시도)
            try:
                self.df = pd.read_csv(self.file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                try:
                    self.df = pd.read_csv(self.file_path, encoding='cp949')
                except UnicodeDecodeError:
                    self.df = pd.read_csv(self.file_path, encoding='latin-1')
        elif file_ext in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")
        
        # 빈 값들을 None으로 통일
        self.df = self.df.replace(['', ' ', '  '], None)
        
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """로드된 DataFrame 반환"""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load() 메서드를 먼저 호출하세요.")
        return self.df

