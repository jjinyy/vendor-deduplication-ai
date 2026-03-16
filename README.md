# vendor-deduplication-ai

Multilingual supplier deduplication pipeline using blocking, fuzzy matching, and embeddings.
Built to solve a problem that simple ID matching can't handle.

---

## Background

~**50K** global supplier records. Duplicate detection via ID matching wasn't an option.

Overseas vendors don't share a universal identifier like Korea's business registration number.
Tax ID systems vary by country — format, availability, and reliability all differ.
Often the data simply isn't there.

So Tax ID was used only as a secondary signal.
Instead, built a scoring model that combines company name, address,
and handled materials to make the call.

---

## Pipeline
```
Source data (~50K records)
    ↓
Step 1: Blocking (candidate compression)
  — Reduces comparisons from 151M → tens of thousands
  — Pre-filter by country, industry, etc.
    ↓
Step 2: ANN (Approximate Nearest Neighbor)
  — Embedding-based similar candidate search
    ↓
Step 3: Scoring (composite score)
  — Company name similarity (raw + romanized + semantic embedding)
  — Address similarity
  — Handled materials overlap
  — Tax ID (secondary signal only)
    ↓
Step 4: Verification & cleansing
  — Extract candidates above threshold → verify → merge
```

---

## Multilingual detection

Same company registered under different languages — detected.
```
"삼성전자" ↔ "Samsung Electronics"
"阿里巴巴集团" ↔ "Alibaba Group" ↔ "알리바바"
"株式会社ソニー" ↔ "Sony Corporation"
```

Chinese → Pinyin conversion, Japanese → Romaji conversion before comparison.
Semantic embeddings handle cross-language similarity beyond character matching.

---

## False positive prevention

Reducing wrong matches mattered as much as finding right ones.

- **Industry keyword comparison**: prevents matching construction firms with utility companies
- **Blocking key improvement**: city name alone doesn't create a candidate pair
- **Embedding weight adjustment**: prevents location-driven false matches
- **Address similarity cap**: limits score when only city name matches

---

## Why this design

**Why a 3-stage hybrid pipeline**
Pairwise comparison across 50K records = ~**151M comparisons**.
Blocking cuts that down to tens of thousands.
ANN handles embedding search efficiently.
Final scoring combines multiple signals for accuracy.

**How it evolved**
```
Start: basic N² comparison, no multilingual support
    ↓
Performance: Blocking + RapidFuzz — comparisons drop drastically
    ↓
Multilingual: Romanization added (Pinyin, Romaji, Hangul)
    ↓
Accuracy: semantic embeddings added (hybrid version)
    ↓
Precision: false positive prevention — industry keywords, address caps
```

---

## Results

- Dataset: ~**50K** global supplier records
- Detection accuracy: **95%+** (final), **92–96%** (hybrid version)
- Comparison reduction: **151M → tens of thousands**
- Duplicate rate detected: ~**30%** of total dataset
- Data cleansing completed → supplier master reliability improved
- Replaced manual verification with data-driven detection workflow

---

## Stack

`Python` `SentenceTransformers` `Multilingual Embeddings`
`RapidFuzz` `ANN` `pandas` `scikit-learn`
`Pinyin` `Romaji` `Fuzzy Matching`

---

## Input / Output

| Type | Location | Description |
|------|----------|-------------|
| **Input** | **`data/`** (preferred) or project root | Source CSV/Excel (e.g. `bio_vendor.csv`, `FOOD 중복업체.xlsx`). Scripts read from `data/` if present. |
| **Output** | **`output/`** | Logs, checkpoints, intermediate and final Excel. **Not tracked in Git** (see `.gitignore`). |

- Korean pipeline: `python run_hybrid_korean.py "data/FOOD 중복업체.xlsx"`

---

## Structure
```
├── run_hybrid.py                  # Overseas vendors (bio_vendor.csv)
├── run_hybrid_korean.py           # Korean vendors (공급업체코, 공급업체명, 주소)
├── src/                           # Core package
│   ├── data_loader.py             # Overseas data loader
│   ├── data_loader_korean.py      # Korean data loader
│   ├── duplicate_detector_hybrid.py
│   ├── cheap_gate.py
│   ├── ann_index.py
│   └── merger.py
├── data/                          # Input (reference) data
├── docs/                          # Documentation
├── output/                        # Generated results (untracked)
├── requirements.txt
└── README.md
```

---

## 기준정보(입력) vs 결과(출력)

| 구분 | 폴더 | 설명 |
|------|------|------|
| **기준정보 (입력)** | **`data/`** (권장) 또는 루트 | `bio_vendor.csv`, `FOOD 중복업체.xlsx` 등 원본 CSV/Excel. 실행 시 `data/`에 있으면 그곳에서 읽고, 없으면 루트에서 읽음. |
| **결과 (출력)** | **`output/`** | 로그, 체크포인트, 중간결과, 병합 결과 Excel. 실행 시 **모두 `output/`에만** 저장됨. |

- **CSV/Excel 기준정보** → **`data/`** 폴더로 옮기세요. (목적별 정리)
- **결과 파일** → 실행하면 **항상 `output/`** 에만 생성됩니다.
- **`output/` 폴더** → `.gitignore`에 포함되어 있어 **Git에는 올라가지 않습니다**. (로그·체크포인트·결과 Excel은 로컬에만 유지)
- 한국 업체: `python run_hybrid_korean.py "data/FOOD 중복업체.xlsx"` 처럼 경로 지정 가능.

## 진행 상황 확인

로그와 결과는 **`output/`** 폴더에서 확인하세요.

```bash
# Windows PowerShell
Get-Content output/process_log_hybrid.txt -Tail 50 -Encoding UTF8
```

## 데이터 구조

프로젝트는 다음 컬럼들을 인식합니다:

- **구매조**: 구매조 코드
- **공급업체코드**: 공급업체 고유 코드
- **Land**: 국가 코드
- **공급업체명**: 공급업체 이름
- **생성일**: 데이터 생성일
- **CITY1, CITY2**: 도시 정보
- **STREET**: 거리명
- **HOUSE_NUM1, HOUSE_NUM2**: 건물 번호
- **STR_SUPPL1, STR_SUPPL2, STR_SUPPL3**: 추가 주소 정보

## 중복 판단 기준

1. **공급업체명 유사도**: 다국어 유사도 계산
   - 원본 문자열 비교
   - 로마자화 버전 비교 (중국어→Pinyin, 일본어→Romaji)
   - 의미 기반 임베딩 비교 (하이브리드 버전)

2. **주소 정보 유사도**: 주소 필드들의 유사도

3. **국가 코드 일치**: 같은 국가 내에서 우선 비교

4. **다국어 표기 동일 업체 판단**: 같은 업체의 다른 언어 표기도 탐지

## 출력 결과

결과 파일에는 다음 컬럼이 포함됩니다:

- `그룹번호`: 같은 업체로 판단된 행들은 같은 그룹번호를 가짐
- `_duplicate_count`: 중복 그룹의 행 수 (중복이 아닌 행은 None)

**중요**: 모든 원본 행이 유지되며, 행이 삭제되지 않습니다. 같은 그룹번호를 가진 행들이 같은 업체로 판단된 것입니다.

## 프로젝트 구조 (목적별 정리)

```
duplCkSupply/
├── docs/                            # 문서
│   ├── 결과_해석_가이드.md          # 결과 컬럼·해석 방법
│   ├── 프로젝트_종합_문서.md        # 시스템 종합 설명
│   └── 한국업체_구조_설명.md        # 한국 업체 입력·실행 구조
├── data/                            # 기준정보(입력) CSV/Excel — 여기로 옮기기
│   ├── bio_vendor.csv               # 해외 업체 기준정보
│   └── FOOD 중복업체.xlsx           # 한국 업체 기준정보
├── output/                          # 결과 전용 (로그·체크포인트·중간·병합 Excel)
│   ├── process_log_hybrid.txt
│   ├── process_log_hybrid_korean.txt
│   ├── checkpoint_ann.pkl / checkpoint_ann_korean.pkl
│   ├── 중간결과_*개그룹_*.xlsx
│   └── *_merged_hybrid_*.xlsx
├── run_hybrid.py                    # 해외 업체 실행 (bio_vendor.csv 등)
├── run_hybrid_korean.py             # 한국 업체 실행 (FOOD 중복업체.xlsx 등)
├── src/                             # 코어 모듈 (구조화)
│   ├── __init__.py
│   ├── data_loader.py               # 해외 데이터 로더
│   ├── data_loader_korean.py        # 한국 데이터 로더 (공급업체코, 공급업체명, 주소)
│   ├── duplicate_detector_hybrid.py # 중복 탐지 메인 로직
│   ├── cheap_gate.py                # STEP1 필터
│   ├── ann_index.py                 # ANN 인덱스
│   └── merger.py                    # 결과 병합·그룹번호 부여
├── requirements.txt
└── README.md
```
