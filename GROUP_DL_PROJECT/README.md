# Nepali Sign Language (NSL) Alphabet Recognition

**CSC60904 Deep Learning - Group Assignment (Track T3) - "AI for the Himalayas 2026"**

A supervised deep-learning system that recognises Nepali Sign Language alphabet
hand-signs from images, to support accessibility for the Deaf community in Nepal.

## Team
| Name | Role |
|------|------|
| _member 1_ | Data Engineer |
| _member 2_ | ML Engineer |
| _member 3_ | Evaluation & Ethics Lead |

## Repository structure
```
.
├── data/         # dataset (not committed - see data/README.md for the link)
├── notebooks/    # NSL_pipeline.ipynb - full pipeline end to end
├── src/          # reusable python modules (extracted from the notebook)
├── models/       # saved .keras models + label_map.json
├── results/      # figures, metrics.json, model_comparison.csv, training logs
├── docs/         # report, meeting minutes, peer evaluations
├── app.py        # Streamlit demo
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt

# 1) train (writes models/ and results/)
jupyter notebook notebooks/NSL_pipeline.ipynb    # run all cells

# 2) demo the trained model
streamlit run app.py
```
Set the dataset path in the notebook's Configuration cell (`DATA_ROOT`).

## Dataset attribution
Birat Poudel, Satyam Ghimire, Sijan Bhattarai, Saurav Bhandari (2025).
*Nepali Sign Language Character Dataset* (MIT License).
Benchmark reference: Poudel et al., "Nepali Sign Language Characters Recognition:
Dataset Development and Deep Learning Approaches", KEC Journal of Science and
Engineering (2026) / arXiv:2510.11243.

## Models
1. Baseline CNN (from scratch).
2. MobileNetV2 transfer learning (feature-extraction + fine-tuning).
