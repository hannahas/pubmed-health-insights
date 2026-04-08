# PubMed Health Insights: TCR Repertoire Sequencing Literature Analysis

An end-to-end NLP and machine learning pipeline that automatically fetches, structures, and classifies biomedical research abstracts using the PubMed API, Claude LLM, and scikit-learn.

---

## Motivation

The biomedical literature grows faster than any researcher can track. For a field like TCR repertoire sequencing — which spans immunology, oncology, infectious disease, and computational biology — keeping up with what kinds of research are being published, what diseases are being studied, and where the field is heading is a real challenge.

This project asks: **can we use LLMs and classical ML to automatically extract structured insight from unstructured biomedical text at scale?**

The answer turns out to be yes — and the results surface some genuinely interesting patterns in where the field is going.

---

## Pipeline Overview

**Step 1 — Data ingestion (`scripts/fetch_abstracts.py`)**  
Queries the NCBI E-utilities API for a given search term and fetches up to 200 abstracts with titles, PMIDs, and publication years. No API key required for basic access.

**Step 2 — LLM feature extraction (`scripts/extract_features.py`)**  
Passes each abstract to Claude (claude-haiku) with a structured prompt, extracting six fields per paper:
- `study_type` — clinical, basic_research, computational, or review
- `sample_size` — number of human subjects if mentioned
- `technology` — sequencing platform or analysis method
- `disease_focus` — primary disease or condition
- `key_finding` — one-sentence summary of the main result
- `clinical_relevance` — high, medium, or low

**Step 3 — ML classification (`scripts/train_classifier.py`)**  
Trains a logistic regression classifier on TF-IDF vectorized abstracts to predict study type. Evaluates with a held-out test set and produces a confusion matrix and per-class performance metrics.

---

## Key Findings

**The TCR repertoire literature is overwhelmingly oncology-focused.**  
Cancer and its subtypes (NSCLC, TNBC, colorectal, glioblastoma, melanoma) dominate the disease landscape. When cancer-related entries are combined, they account for the majority of disease-focused papers — dwarfing all other conditions.

**COVID-19 remains a significant presence.**  
Despite being several years post-pandemic, COVID-19 is the second most common disease focus in the dataset, reflecting the lasting influence of pandemic-era immune response research on the TCR repertoire field.

**The field skews clinical, but computational research is growing.**  
86 of 200 papers (43%) are clinical studies involving human subjects. Computational papers — focused on HLA binding prediction, TCR-pMHC specificity, and ML-based repertoire analysis — account for 19% of the literature and represent the fastest-growing segment.

**An LLM + classical ML pipeline can learn real biology.**  
The logistic regression classifier reached 68% accuracy on a 4-class problem with only 200 training samples. More importantly, the most predictive words per class are scientifically meaningful:
- *Basic research:* `mice`, `cd8`, `cd4`, `epitopes` — animal models and T cell biology
- *Clinical:* `patients`, `blood`, `lung`, `breast` — human subjects and disease
- *Computational:* `hla`, `prediction`, `pmhc`, `sequences` — bioinformatics tools
- *Review:* `advances`, `approaches`, `therapeutic` — synthesis language

The model is not memorizing noise — it's capturing real distinctions between research paradigms.

---

## How to Run

**1. Clone the repo and set up the environment**
```bash
git clone https://github.com/hannahas/pubmed-health-insights.git
cd pubmed-health-insights
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Add your Anthropic API key**  
Create a `.env` file in the project root:
ANTHROPIC_API_KEY=xxxxxxxxx

**3. Fetch abstracts**
```bash
python3 scripts/fetch_abstracts.py
```

**4. Extract structured features with Claude**
```bash
python3 scripts/extract_features.py
```

**5. Train and evaluate the classifier**
```bash
python3 scripts/train_classifier.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Data source | NCBI PubMed E-utilities API |
| LLM extraction | Anthropic Claude (claude-haiku) |
| Data manipulation | pandas |
| Text vectorization | scikit-learn TF-IDF |
| Classification | scikit-learn Logistic Regression |
| Visualization | matplotlib, seaborn |
| Environment | Python 3.14, venv |
| Version control | Git / GitHub |

---

## Results

| Metric | Value |
|---|---|
| Overall accuracy | 68% |
| Clinical F1 | 0.74 |
| Basic research F1 | 0.69 |
| Computational F1 | 0.40 |
| Training samples | 158 |
| Test samples | 40 |

*Note: "review" class performance is unreliable due to only 6 samples in the dataset.*

---

## A note on ground truth and model design

This project involves two very different kinds of classifiers, and understanding the distinction is important for interpreting the results honestly.

**Claude (ground truth labeler)** reads the full title and abstract and makes a holistic judgment — the same way a human expert would. It understands sentence structure, context, and nuance. When it labels a paper as "clinical", it has genuinely read and understood the abstract.

**The ML classifier** reduces each abstract to a weighted frequency count of the 500 most informative words (TF-IDF), with no understanding of word order or meaning. It learned that words like `patients`, `blood`, and `lung` correlate with clinical labels — but it is pattern matching, not comprehending.

This means the classifier is not predicting the "true" study type — it is learning to **replicate Claude's labeling behavior from text features alone**. This is a useful and practical outcome: once trained, the classifier can label thousands of new abstracts instantly and cheaply, without an API call for each one.

Label quality was assessed qualitatively by reviewing the top predictive words per class. The fact that these words align closely with domain expectations — `mice`, `cd8`, `epitopes` for basic research; `hla`, `pmhc`, `prediction` for computational — suggests the Claude-generated labels are scientifically coherent and the classifier has learned meaningful signal rather than noise.

A natural next step would be to replace TF-IDF + logistic regression with a fine-tuned language model such as BioBERT, which encodes contextual meaning the way Claude does. This would likely push accuracy well above 80% and close the gap between how the ground truth was generated and how predictions are made.

---

## Next Steps

- [ ] Add cross-validation for more robust performance estimates
- [ ] Predict `clinical_relevance` as a second classification target
- [ ] Expand dataset to 500+ abstracts across multiple search terms
- [ ] Build a Streamlit app for interactive abstract classification
- [ ] Write a companion blog post with full methodology and findings

---

## Author

Alexander Hannah, PhD  
Computational biologist and data scientist  
[github.com/hannahas](https://github.com/hannahas)