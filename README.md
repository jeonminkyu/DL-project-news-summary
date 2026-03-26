# 📰 News Abstractive Summarization — Seq2Seq + Attention

AIFFEL Exploration 2 — Seq2Seq(LSTM) + Additive Attention 모델을 구현하여 뉴스 기사를 자동 요약하고, Extractive 방식(TextRank)과 성능을 비교하는 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목적 | 뉴스 본문(text) → 헤드라인(headlines) 자동 요약 |
| 데이터셋 | [News Summary Dataset](https://github.com/sunnysai12345/News_Summary) (`news_summary_more.csv`, 약 98,000개 샘플) |
| 프레임워크 | TensorFlow / Keras |
| 요약 방식 | Abstractive (Seq2Seq + Attention) vs Extractive (summa / TextRank) |
| 평가 지표 | BLEU, ROUGE |
| 언어 | Python 3 |

---

## 🗂️ 주요 파일 구조

```
news_summary/
├── data/
│   └── news_summary_more.csv   # 뉴스 원문 및 헤드라인 데이터
└── aiffelproject_exploration2-try2.ipynb  # 메인 노트북
```

---

## ⚙️ 주요 구현 내용

### 1. 데이터 전처리
- 중복 제거 및 결측치 처리
- 축약어 정규화 사전 적용 (e.g., `don't` → `do not`)
- HTML 태그 제거, 특수문자 제거, 소문자 변환
- 불용어(stopwords) 제거 — 본문(text)에만 적용, 요약(headlines)은 미적용
- 텍스트 최대 길이 설정: **본문 45단어, 요약 12단어** (길이 분포 시각화 후 결정)
- 디코더 입력/출력에 `sostoken` / `eostoken` 추가

### 2. 정수 인코딩 & 패딩
- 희귀 단어 제거 (본문: 빈도 7 미만, 요약: 빈도 6 미만)
- 인코더 단어 집합 크기: **20,000** / 디코더 단어 집합 크기: **10,000**
- 훈련 / 테스트 데이터 분리: **8 : 2**
- `pad_sequences`로 최대 길이에 맞춰 post-padding 적용

### 3. 모델 아키텍처 — Seq2Seq + Additive Attention

```
Encoder
  └── Embedding(src_vocab=20000, dim=128)
  └── LSTM 1 (hidden=256, dropout=0.4, return_sequences=True)
  └── LSTM 2 (hidden=256, dropout=0.4, return_sequences=True)
  └── LSTM 3 (hidden=256, dropout=0.4, return_sequences=True)
        └── encoder_output, state_h, state_c 반환

Decoder
  └── Embedding(tar_vocab=10000, dim=128)
  └── LSTM (hidden=256, dropout=0.4, initial_state=[state_h3, state_c3])
  └── AdditiveAttention([decoder_outputs, encoder_output3])
  └── Concatenate([decoder_outputs, attn_out])
  └── Dense(tar_vocab, activation='softmax')
```

| 하이퍼파라미터 | 값 |
|---|---|
| Embedding Dimension | 128 |
| Hidden Size (LSTM) | 256 |
| Encoder LSTM 층 수 | 3 |
| Dropout | 0.4 |
| Batch Size | 256 |
| Max Epochs | 50 (EarlyStopping patience=2) |
| Optimizer | RMSProp |
| Loss | Sparse Categorical Crossentropy |

### 4. 인퍼런스
- 학습 완료 후 **인코더 모델과 디코더 모델을 분리**하여 인퍼런스 구성
- `sostoken`부터 시작하여 `eostoken` 또는 최대 길이 도달 시 생성 종료

---

## 📊 평가 방법

| 방법 | 라이브러리 | 설명 |
|------|-----------|------|
| **BLEU** | `nltk.translate.bleu_score` | n-gram 기반 생성 요약 품질 평가 |
| **ROUGE** | `rouge` | F1-score, Precision, Recall 반환 |
| **단어 빈도 분석** | `nltk.FreqDist` | 실제 요약 vs 예측 요약의 주요 단어 비교 |

---

## 🔄 Abstractive vs Extractive 비교

| 방식 | 방법 | 특징 |
|------|------|------|
| **Abstractive** | Seq2Seq + Attention (직접 구현) | 새로운 문장을 생성하여 요약 |
| **Extractive** | `summa` 라이브러리 (TextRank) | 원문에서 핵심 문장을 선택하여 요약 |

두 방식의 예측 요약과 실제 헤드라인을 나란히 비교하고, 주요 단어 빈도를 DataFrame으로 시각화합니다.

---

## 🛠️ 실행 환경

```bash
pip install tensorflow nltk pandas numpy matplotlib beautifulsoup4 summa rouge
```

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 📈 성능 향상 아이디어

- 더 많은 학습 데이터 및 에폭 수 증가
- Bidirectional LSTM 적용
- Beam Search 디코딩 적용
- KoBERT, T5 등 사전학습 모델 활용 (Transfer Learning)
- 하이퍼파라미터 튜닝 (hidden size, embedding dim 등)

---

## 📝 회고

> 루브릭에 맞추어 끝까지 프로젝트를 완성하기 위해 노력했습니다.
> 모델의 성능이 기대에 미치지 못했으며, 실제 요약과 예측 요약 사이에 큰 차이가 있었습니다.
> 향후 모델 구조와 학습 방법을 개선하여 성능을 높이는 것을 목표로 합니다.

---

## 📚 참고

- [News Summary Dataset](https://github.com/sunnysai12345/News_Summary)
- [summa (TextRank)](https://github.com/summanlp/textrank)
- [Bahdanau Attention 논문](https://arxiv.org/abs/1409.0473)
- AIFFEL Exploration 2
