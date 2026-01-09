"""
중복 데이터 병합 모듈
중복 그룹을 하나의 대표 행으로 병합합니다.
"""
import pandas as pd
from typing import List, Dict
import numpy as np


class DataMerger:
    """중복 데이터를 병합하는 클래스"""
    
    def __init__(self, merge_strategy: str = 'best'):
        """
        Args:
            merge_strategy: 병합 전략
                - 'best': 가장 완전한 데이터를 가진 행 선택
                - 'first': 첫 번째 행 선택
                - 'latest': 최신 생성일을 가진 행 선택
        """
        self.merge_strategy = merge_strategy
    
    def calculate_completeness_score(self, row: pd.Series) -> float:
        """행의 데이터 완전도 점수 계산 (0.0 ~ 1.0)"""
        important_fields = [
            '공급업체명', 'Land', 'CITY1', 'STREET', 
            'STR_SUPPL1', 'STR_SUPPL2', 'HOUSE_NUM1'
        ]
        
        filled_count = 0
        for field in important_fields:
            if field in row.index:
                value = row[field]
                if pd.notna(value) and str(value).strip() != '':
                    filled_count += 1
        
        return filled_count / len(important_fields) if important_fields else 0.0
    
    def merge_group(self, df: pd.DataFrame, group_indices: List[int]) -> pd.Series:
        """
        중복 그룹을 하나의 행으로 병합
        
        Args:
            df: 원본 DataFrame
            group_indices: 병합할 행들의 인덱스 리스트
            
        Returns:
            병합된 행 (Series)
        """
        group_rows = df.iloc[group_indices].copy()
        
        if self.merge_strategy == 'best':
            # 가장 완전한 데이터를 가진 행 선택
            completeness_scores = group_rows.apply(
                self.calculate_completeness_score, axis=1
            )
            best_idx = completeness_scores.idxmax()
            merged_row = group_rows.loc[best_idx].copy()
            
        elif self.merge_strategy == 'latest':
            # 최신 생성일을 가진 행 선택
            if '생성일' in group_rows.columns:
                # 날짜 파싱 시도
                dates = []
                for idx in group_indices:
                    date_val = df.loc[idx, '생성일']
                    if pd.notna(date_val):
                        try:
                            # 다양한 날짜 형식 처리
                            if isinstance(date_val, str):
                                date_val = pd.to_datetime(date_val, errors='coerce')
                            dates.append((idx, date_val))
                        except:
                            dates.append((idx, None))
                
                if dates:
                    # 유효한 날짜가 있는 경우 최신 것 선택
                    valid_dates = [(idx, dt) for idx, dt in dates if pd.notna(dt)]
                    if valid_dates:
                        latest_idx = max(valid_dates, key=lambda x: x[1])[0]
                        merged_row = group_rows.loc[latest_idx].copy()
                    else:
                        merged_row = group_rows.iloc[0].copy()
                else:
                    merged_row = group_rows.iloc[0].copy()
            else:
                merged_row = group_rows.iloc[0].copy()
        else:  # 'first'
            merged_row = group_rows.iloc[0].copy()
        
        # 빈 필드들을 다른 행의 값으로 채움
        for col in merged_row.index:
            if pd.isna(merged_row[col]) or str(merged_row[col]).strip() == '':
                for idx in group_indices:
                    if idx != merged_row.name:
                        other_value = df.loc[idx, col]
                        if pd.notna(other_value) and str(other_value).strip() != '':
                            merged_row[col] = other_value
                            break
        
        # 공급업체코드는 여러 개를 쉼표로 구분하여 저장
        if '공급업체코드' in merged_row.index:
            codes = [str(df.loc[idx, '공급업체코드']) 
                    for idx in group_indices 
                    if pd.notna(df.loc[idx, '공급업체코드'])]
            if len(codes) > 1:
                merged_row['공급업체코드'] = ', '.join(set(codes))
        
        # 중복 그룹 정보 추가
        merged_row['_duplicate_count'] = len(group_indices)
        merged_row['_merged_from_indices'] = ', '.join(map(str, group_indices))
        
        return merged_row
    
    def merge_duplicates(self, df: pd.DataFrame, duplicate_groups: List[List[int]]) -> pd.DataFrame:
        """
        중복 그룹들을 병합하여 새로운 DataFrame 생성
        
        Args:
            df: 원본 DataFrame
            duplicate_groups: 중복 그룹 리스트
            
        Returns:
            병합된 DataFrame
        """
        # 중복 그룹에 포함된 모든 인덱스
        duplicate_indices = set()
        for group in duplicate_groups:
            duplicate_indices.update(group)
        
        # 중복이 아닌 행들
        non_duplicate_rows = df[~df.index.isin(duplicate_indices)].copy()
        
        # 각 중복 그룹을 병합
        merged_rows = []
        for group in duplicate_groups:
            merged_row = self.merge_group(df, group)
            merged_rows.append(merged_row)
        
        # 병합된 행들을 DataFrame으로 변환
        if merged_rows:
            merged_df = pd.DataFrame(merged_rows)
            # 인덱스 재설정
            merged_df.reset_index(drop=True, inplace=True)
            
            # 중복이 아닌 행들과 병합
            result_df = pd.concat([non_duplicate_rows, merged_df], ignore_index=True)
        else:
            result_df = non_duplicate_rows.copy()
        
        return result_df

