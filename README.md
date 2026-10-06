# Kinyarwanda News Topic Classification

Text classification for Kinyarwanda: news articles are assigned to one of 14 topics.
Capstone project comparing classical baselines, a BiLSTM, and a fine-tuned AfriBERTa transformer, with error analysis and a web app.

## Project links
- GitHub repository: https://github.com/Seziberagabriel/kinyarwanda-news-classifier
- Live system (Streamlit): <ADD STREAMLIT LINK>
- Demo video: <ADD VIDEO LINK>
- Trained model (Hugging Face Hub): https://huggingface.co/Seziberagabriel/kinyarwanda-news-classifier
- Run everything on Colab: `notebooks/run_on_colab.ipynb`

## Problem
Given a Kinyarwanda news article (title + content), predict its topic.

## Dataset
KINNEWS (Niyongabo et al., 2020), Hugging Face `andreniyongabo/kinnews_kirnews`, config `kinnews_cleaned`, MIT license.
14 classes: politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, relationship.

**Data quality findings** (`src/data_quality.py`, `results/data_quality.txt`): the raw training set (17,014 rows) contains 7,939 extra copies of duplicate articles, 109 texts with conflicting labels, and 2,226 of the 4,254 test rows have exact copies in the training set. I therefore removed duplicates and test/train overlaps (`src/prepare_data.py`).
Final splits: **train 8,167 / validation 908 / test 2,008**. Classes are imbalanced (e.g. tourism has only 8 test articles), so macro-F1 is reported alongside accuracy.

## Methods
| Step | Script | Description |
|---|---|---|
| 1 | `src/prepare_data.py` | download, clean, de-duplicate, split, statistics |
| 2 | `src/data_quality.py` | diagnose duplicates / leakage / conflicting labels |
| 3 | `src/baseline.py` | TF-IDF (1-2 grams, 50k features) + Logistic Regression / Naive Bayes |
| 4 | `src/bilstm.py` | BiLSTM with max pooling, embeddings trained from scratch |
| 5 | `src/finetune.py` | fine-tune `castorini/afriberta_base` (optional class-weighted loss) |
| 6 | `src/error_analysis.py` | confusion matrix, error rate by length, error samples |
| 7 | `src/upload_to_hub.py` | publish the model to the Hugging Face Hub |

## Results (test set, 2,008 articles)
| Model | Config | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---|---|---|---|---|
| tfidf_logreg | TF-IDF 1-2gram 50k, LogReg C=10 | 0.802 | 0.804 | 0.683 | 0.722 |
| tfidf_nb | TF-IDF 1-2gram 50k, MultinomialNB alpha=0.1 | 0.758 | 0.832 | 0.529 | 0.579 |
| bilstm | emb=200, hid=128, maxlen=256, epochs=8, lr=0.001, w2v=False | 0.708 | 0.705 | 0.522 | 0.524 |
| afriberta_base | castorini/afriberta_base, lr=3e-05, epochs=3, max_len=256, batch=16, n_train=8167, class_weights=False | 0.778 | 0.684 | 0.697 | 0.676 |
| afriberta_base_len128 | castorini/afriberta_base, lr=3e-05, epochs=3, max_len=128, batch=16, n_train=8167, class_weights=False | 0.758 | 0.654 | 0.654 | 0.641 |
| afriberta_base_lr2e5 | castorini/afriberta_base, lr=2e-05, epochs=4, max_len=256, batch=16, n_train=8167, class_weights=False | 0.778 | 0.687 | 0.704 | 0.681 |

The TF-IDF + Logistic Regression baseline outperformed the fine-tuned transformer; see the report for analysis.
Per-class reports: `results/report_afriberta_base.txt`. Confusion matrix: `results/confusion_afriberta_base.png`.

## How to run
```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/data_quality.py
python src/baseline.py
python src/bilstm.py --name bilstm --epochs 8
python src/finetune.py --model castorini/afriberta_base --name afriberta_base --lr 3e-5 --epochs 3 --max_len 256
python src/error_analysis.py --name afriberta_base
```
A GPU is needed for fine-tuning; the easiest way is `notebooks/run_on_colab.ipynb` on a free Colab T4.

Web app (Streamlit):
```bash
pip install -r app/requirements.txt streamlit
streamlit run app/streamlit_app.py
```
A Gradio version is also provided (`app/gradio_app.py`, `app/requirements_gradio.txt`).

## Limitations
News domain only; articles truncated to 256 tokens; class imbalance and label ambiguity (e.g. COVID-19 news labelled as health, politics or sport); small de-duplicated training set; single dataset.

## References
- Niyongabo, R. A., Qu, H., Kreutzer, J., Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi. arXiv:2010.12174.
- Ogueji, K., Zhu, Y., Lin, J. (2021). Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-Resource Languages. (AfriBERTa)
- Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing.
(Verify every reference against the original source before submitting.)
