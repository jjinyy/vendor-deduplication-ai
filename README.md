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

## Additional documentation

- Detailed process and architecture: see `docs/프로젝트_종합_문서.md`
- Result columns and interpretation: see `docs/결과_해석_가이드.md`
- Korean vendor input format & run guide: see `docs/한국업체_구조_설명.md`
