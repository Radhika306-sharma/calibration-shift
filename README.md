# Classifier Calibration Under Distribution Shift

**When do confidence scores remain probabilities?**

This repository contains the code, data, results, and manuscript for a Probability & Statistics research project studying how classifier calibration (the correspondence between predicted confidence and true probability of correctness) degrades as train/test distributions diverge.

## Research question

When a classifier is trained on one data distribution and evaluated on a shifted one, does its predicted confidence remain a statistically valid estimate of correctness — and does this degrade differently across model families, in a way that's statistically associated with the magnitude of the shift?

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, binary target: churn). Two natural, business-meaningful drift axes were used to construct shift levels:

- **Contract type** (month-to-month / one-year / two-year)
- **Tenure** (months as a customer, 0–72)

| Test set | Population |
|---|---|
| Base (train) | Month-to-month, tenure 0–12 months |
| Shift 0 (in-distribution) | Held-out month-to-month, tenure 0–12 |
| Shift 1 (mild) | Month-to-month, tenure 12–24 |
| Shift 2 (moderate) | One-year contract, tenure 24–48 |
| Shift 3 (strong) | Two-year contract, tenure 48–60 |
| Shift 4 (severe) | Two-year contract, tenure 60–72 |

Shift magnitude is independently quantified via Population Stability Index (PSI) and KL-divergence on the numeric features.

## Models

Logistic Regression, Gaussian Naive Bayes, Random Forest, SVM (RBF kernel), XGBoost — all trained on the base distribution with a shared preprocessing pipeline, then evaluated across shift levels.

## Metrics & statistical tests

- **Accuracy**, **Brier score**, **Expected Calibration Error (ECE)** per model per shift level
- **Paired bootstrap t-tests** and **Wilcoxon signed-rank tests** between every pair of models at every shift level
- **Pearson / Spearman correlation** between shift magnitude (PSI) and ECE

## Repository structure

```
.
├── src/
│   └── calibration_shift_pipeline.py   # end-to-end pipeline: data prep, training, metrics, stats, plots
├── notebooks/
│   └── calibration_analysis.ipynb      # exploratory / walkthrough notebook
├── data/
│   └── README.md                       # dataset description and source
├── results/
│   ├── results_summary.csv             # accuracy, Brier score, ECE per model x shift level
│   ├── pairwise_model_comparisons.csv  # paired t-test / Wilcoxon results
│   └── shift_vs_ece_correlation.csv    # PSI-vs-ECE correlation results
├── figures/
│   └── reliability_diagrams.png        # reliability diagrams, all models x all shift levels
├── paper/
│   └── paper.pdf                       # manuscript
├── telco.csv                           # raw dataset
├── requirements.txt
└── README.md
```

## Key finding

Calibration degrades unevenly across model families as distribution shift increases. Logistic Regression's ECE stays lowest or near-lowest at every shift level (0.024 → 0.031 from Shift 0 to Shift 4), even as its accuracy changes substantially. SVM shows the opposite pattern: accuracy collapses under severe shift (0.69 → 0.03) while its confidence stays high, producing sharply increasing ECE (0.055 → 0.478) — the model becomes confidently wrong rather than appropriately uncertain. Pooled across all models, shift magnitude (PSI) and ECE are significantly correlated (Pearson r = 0.51, p = 0.009).

## Reproducing the results

```bash
pip install -r requirements.txt
python3 src/calibration_shift_pipeline.py
```

This downloads/loads the dataset, runs the full pipeline, and writes all CSVs in `results/` and the figure in `figures/`.

## References

- Ovadia, Y. et al. (2019). *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* NeurIPS 32.
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning.* ICML.
- Ng, A. Y. & Jordan, M. I. (2002). *On Discriminative vs. Generative Classifiers.* NeurIPS 14.
- Quiñonero-Candela, J. et al., eds. (2009). *Dataset Shift in Machine Learning.* MIT Press.