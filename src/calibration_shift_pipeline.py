"""
Classifier Calibration Under Distribution Shift: Experimental Pipeline
========================================================================

Dataset: IBM Telco Customer Churn (7,043 customers, binary target = Churn).
Chosen because it has two natural, business-meaningful "drift axes":
    - Contract type (Month-to-month / One year / Two year)
    - tenure (months as a customer, 0-72)
Both are known to correlate strongly with churn behavior, so splitting on
them produces test sets whose P(X) AND P(Y|X) both drift relative to the
training distribution -- exactly the "dataset shift" setting the paper is
about (as opposed to a shift that only moves the label balance).

Shift design
------------
BASE (training distribution): Month-to-month contracts, tenure 0-12 months.
  This is the largest, "newest customer" segment.
Shift 0 (in-distribution control): held-out Month-to-month, tenure 0-12.
Shift 1 (mild):     Month-to-month, tenure 12-24.
Shift 2 (moderate): One year contract, tenure 24-48.
Shift 3 (strong):   Two year contract, tenure 48-60.
Shift 4 (severe):   Two year contract, tenure 60-72.
Moving from Shift 0 -> Shift 4 walks steadily away from the training
population on both the contract axis and the tenure axis, and churn rate
drops steadily as tenure/commitment increases, so this is a real
covariate + prior-probability shift, not an artificial one.

Pipeline steps
--------------
1. Load & clean the data.
2. Carve out BASE train/test and the four shift test sets.
3. Quantify train-vs-test divergence with both PSI and KL-divergence,
   computed on the numeric features (tenure, MonthlyCharges, TotalCharges).
4. Fit a shared preprocessing pipeline (one-hot + scaling) on BASE TRAIN only.
5. Train 5 classifiers on BASE TRAIN: Logistic Regression, Gaussian Naive
   Bayes, Random Forest, SVM (RBF, probability=True), XGBoost.
6. For every (model, shift level) pair compute Accuracy, Brier score, and
   Expected Calibration Error (ECE), and draw a reliability diagram.
7. Use paired bootstrap resampling of each test set to get a sampling
   distribution of ECE per model, enabling:
     - paired t-test and Wilcoxon signed-rank test between every pair of
       models, at every shift level (same bootstrap draws are reused
       across models -> valid pairing).
     - Pearson/Spearman correlation between shift magnitude (PSI) and ECE.
8. Write all numeric results to CSV files and print formatted summary
   tables for direct use in the results section.

Everything below is intentionally verbose in its comments: the goal is for
you to be able to explain every statistical step yourself.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, brier_score_loss
from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------------------------

DATA_PATH = "telco.csv"  # IBM Telco Customer Churn dataset
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

def ensure_data(path: str, url: str) -> None:
    """Download the dataset once if it isn't already sitting next to this script."""
    import os
    if os.path.exists(path):
        return
    import urllib.request
    print(f"Downloading dataset to {path} ...")
    urllib.request.urlretrieve(url, path)

def load_data(path: str) -> pd.DataFrame:
    ensure_data(path, DATA_URL)
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    # TotalCharges is stored as a string with some blank entries (new
    # customers with 0 months tenure) -> coerce to numeric, fill blanks with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df

df = load_data(DATA_PATH)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    c for c in df.columns
    if c not in NUMERIC_FEATURES + ["Churn"]
]
TARGET = "Churn"

# ---------------------------------------------------------------------------
# 2. DEFINE BASE DISTRIBUTION AND SHIFT LEVELS
# ---------------------------------------------------------------------------

def base_mask(d):
    return (d["Contract"] == "Month-to-month") & (d["tenure"] <= 12)

SHIFT_DEFINITIONS = {
    "Shift 1 (mild)":     lambda d: (d["Contract"] == "Month-to-month") & (d["tenure"] > 12) & (d["tenure"] <= 24),
    "Shift 2 (moderate)": lambda d: (d["Contract"] == "One year")       & (d["tenure"] > 24) & (d["tenure"] <= 48),
    "Shift 3 (strong)":   lambda d: (d["Contract"] == "Two year")       & (d["tenure"] > 48) & (d["tenure"] <= 60),
    "Shift 4 (severe)":   lambda d: (d["Contract"] == "Two year")       & (d["tenure"] > 60) & (d["tenure"] <= 72),
}

base_df = df[base_mask(df)].reset_index(drop=True)

# Split the base population itself into a train fold and an in-distribution
# held-out fold. The held-out fold is "Shift 0" -- our control group, which
# should show the *lowest* calibration error for every model since it comes
# from the same distribution the models were fit on.
from sklearn.model_selection import train_test_split
base_train_df, base_holdout_df = train_test_split(
    base_df, test_size=0.30, random_state=RNG_SEED, stratify=base_df[TARGET]
)

TEST_SETS = {"Shift 0 (in-distribution)": base_holdout_df}
for name, mask_fn in SHIFT_DEFINITIONS.items():
    TEST_SETS[name] = df[mask_fn(df)].reset_index(drop=True)

print("Sample sizes:")
print(f"  BASE TRAIN               : {len(base_train_df)}")
for name, d in TEST_SETS.items():
    print(f"  {name:<28}: {len(d)}  (churn rate = {d[TARGET].mean():.3f})")

# ---------------------------------------------------------------------------
# 3. QUANTIFY DISTRIBUTION SHIFT: PSI AND KL-DIVERGENCE
# ---------------------------------------------------------------------------
#
# Both PSI and KL-divergence compare a reference distribution (train) to a
# comparison distribution (test) after binning a continuous variable into
# discrete buckets. We bin using the TRAINING data's decile edges so that
# bin boundaries are fixed and not re-derived from the (possibly shifted)
# test data -- this is standard practice in PSI-based monitoring.

def _bin_edges(train_col: pd.Series, n_bins: int = 10) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train_col, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf  # catch out-of-range test values
    return edges

def _binned_proportions(col: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(col, bins=edges)
    props = counts / counts.sum()
    props = np.clip(props, 1e-6, None)  # avoid log(0) / division by 0
    props = props / props.sum()         # renormalize after clipping
    return props

def population_stability_index(train_col: pd.Series, test_col: pd.Series, n_bins: int = 10) -> float:
    """
    PSI = sum( (test_pct - train_pct) * ln(test_pct / train_pct) )
    Rule of thumb: <0.1 negligible shift, 0.1-0.25 moderate, >0.25 major shift.
    """
    edges = _bin_edges(train_col, n_bins)
    p_train = _binned_proportions(train_col, edges)
    p_test = _binned_proportions(test_col, edges)
    return float(np.sum((p_test - p_train) * np.log(p_test / p_train)))

def kl_divergence(train_col: pd.Series, test_col: pd.Series, n_bins: int = 10) -> float:
    """
    KL(test || train) = sum( test_pct * ln(test_pct / train_pct) )
    Interpreted here as "information lost approximating the shifted test
    distribution with the training distribution" -- i.e. how surprised the
    model's training-time view of the world would be by the test data.
    """
    edges = _bin_edges(train_col, n_bins)
    p_train = _binned_proportions(train_col, edges)
    p_test = _binned_proportions(test_col, edges)
    return float(np.sum(p_test * np.log(p_test / p_train)))

def average_shift_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame, features=NUMERIC_FEATURES) -> dict:
    """Average PSI and KL across the numeric features as one scalar 'shift magnitude' per test set."""
    psis, kls = [], []
    for f in features:
        psis.append(population_stability_index(train_df[f], test_df[f]))
        kls.append(kl_divergence(train_df[f], test_df[f]))
    return {"PSI": float(np.mean(psis)), "KL": float(np.mean(kls))}

shift_magnitude = {}
for name, test_df in TEST_SETS.items():
    shift_magnitude[name] = average_shift_metrics(base_train_df, test_df)

print("\nShift magnitude (averaged across tenure, MonthlyCharges, TotalCharges):")
shift_mag_df = pd.DataFrame(shift_magnitude).T
print(shift_mag_df.round(4))

# ---------------------------------------------------------------------------
# 4. PREPROCESSING (fit on BASE TRAIN only -- never refit on test data)
# ---------------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

X_train = base_train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = base_train_df[TARGET].values

preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)

def transform(d: pd.DataFrame):
    X = d[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return preprocessor.transform(X), d[TARGET].values

# ---------------------------------------------------------------------------
# 5. TRAIN MODELS ON THE BASE DISTRIBUTION
# ---------------------------------------------------------------------------

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RNG_SEED),
    "NaiveBayes": GaussianNB(),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RNG_SEED),
    "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=RNG_SEED),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=RNG_SEED, verbosity=0
    ),
}

fitted_models = {}
for name, model in MODELS.items():
    Xt = X_train_t.toarray() if hasattr(X_train_t, "toarray") and name == "NaiveBayes" else X_train_t
    model.fit(Xt, y_train)
    fitted_models[name] = model
print("\nAll 5 models trained on BASE TRAIN.")

def predict_proba_safe(model, X, model_name):
    """GaussianNB needs a dense array; sparse one-hot output otherwise works for all models here."""
    if model_name == "NaiveBayes" and hasattr(X, "toarray"):
        X = X.toarray()
    return model.predict_proba(X)[:, 1]

# ---------------------------------------------------------------------------
# 6. CALIBRATION METRICS: ACCURACY, BRIER SCORE, EXPECTED CALIBRATION ERROR
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Standard equal-width-bin ECE:
      1. Bin predicted probabilities into n_bins equal-width bins over [0,1].
      2. In each bin, compare the bin's mean predicted probability
         (confidence) to the bin's actual positive rate (accuracy).
      3. ECE = sum over bins of (bin_weight * |confidence - accuracy|).
    A perfectly calibrated model has ECE = 0: predicted probabilities match
    observed frequencies at every confidence level.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)

def reliability_curve(y_true, y_prob, n_bins=10):
    """Returns (mean predicted prob, observed frequency, bin weight) per bin, for plotting."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    confs, accs, weights = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        confs.append(y_prob[mask].mean())
        accs.append(y_true[mask].mean())
        weights.append(mask.sum())
    return np.array(confs), np.array(accs), np.array(weights)

# Store raw (y_true, y_prob) per (model, test set) -- reused for bootstrap tests
predictions = {}  # predictions[test_name][model_name] = (y_true, y_prob)
results_rows = []

for test_name, test_df in TEST_SETS.items():
    X_test_t, y_test = transform(test_df)
    predictions[test_name] = {}
    for model_name, model in fitted_models.items():
        y_prob = predict_proba_safe(model, X_test_t, model_name)
        y_pred = (y_prob >= 0.5).astype(int)
        predictions[test_name][model_name] = (y_test, y_prob)

        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test, y_prob)

        results_rows.append({
            "test_set": test_name,
            "model": model_name,
            "n": len(y_test),
            "PSI": shift_magnitude[test_name]["PSI"],
            "KL": shift_magnitude[test_name]["KL"],
            "accuracy": acc,
            "brier_score": brier,
            "ece": ece,
        })

results_df = pd.DataFrame(results_rows)
results_df.to_csv("results_summary.csv", index=False)

print("\n=== RESULTS TABLE (accuracy / Brier / ECE per model per shift level) ===")
print(results_df.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# RELIABILITY DIAGRAMS: one figure per shift level, all 5 models overlaid
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, len(TEST_SETS), figsize=(4 * len(TEST_SETS), 4), sharey=True)
for ax, (test_name, model_dict) in zip(axes, predictions.items()):
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    for model_name, (y_true, y_prob) in model_dict.items():
        confs, accs, weights = reliability_curve(y_true, y_prob)
        ax.plot(confs, accs, marker="o", markersize=3, linewidth=1, label=model_name)
    ax.set_title(test_name, fontsize=9)
    ax.set_xlabel("Mean predicted probability")
axes[0].set_ylabel("Observed frequency of positive class")
axes[0].legend(fontsize=6, loc="upper left")
plt.tight_layout()
plt.savefig("reliability_diagrams.png", dpi=150)
plt.close()
print("\nSaved reliability_diagrams.png")

# ---------------------------------------------------------------------------
# 7. PAIRED BOOTSTRAP SIGNIFICANCE TESTS
# ---------------------------------------------------------------------------
#
# A single ECE value per (model, shift level) is a point estimate -- it has
# no variance, so a t-test/Wilcoxon test can't be run directly on it.
# To get a *sampling distribution* of ECE that is comparable (paired) across
# models, we bootstrap-resample the SAME test-set row indices B times and
# recompute ECE for every model on each resample. Because every model sees
# the identical set of resampled rows on each iteration, model A's b-th ECE
# and model B's b-th ECE are a legitimate matched pair -> paired t-test and
# Wilcoxon signed-rank test are both valid here.

N_BOOTSTRAP = 300

def bootstrap_ece_distribution(test_name: str, n_boot: int = N_BOOTSTRAP) -> dict:
    """Returns {model_name: np.array of length n_boot of bootstrap ECE values}."""
    model_dict = predictions[test_name]
    n = len(next(iter(model_dict.values()))[0])
    boot_indices = [rng.integers(0, n, size=n) for _ in range(n_boot)]  # same draws reused for every model
    boot_ece = {name: np.empty(n_boot) for name in model_dict}
    for name, (y_true, y_prob) in model_dict.items():
        for b, idx in enumerate(boot_indices):
            boot_ece[name][b] = expected_calibration_error(y_true[idx], y_prob[idx])
    return boot_ece

pairwise_rows = []
for test_name in TEST_SETS:
    boot_ece = bootstrap_ece_distribution(test_name)
    for model_a, model_b in combinations(boot_ece.keys(), 2):
        a_vals, b_vals = boot_ece[model_a], boot_ece[model_b]
        t_stat, t_p = ttest_rel(a_vals, b_vals)
        # Wilcoxon requires at least one non-zero difference
        diffs = a_vals - b_vals
        if np.allclose(diffs, 0):
            w_stat, w_p = np.nan, 1.0
        else:
            w_stat, w_p = wilcoxon(a_vals, b_vals)
        pairwise_rows.append({
            "test_set": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "mean_ece_a": a_vals.mean(),
            "mean_ece_b": b_vals.mean(),
            "paired_t_stat": t_stat,
            "paired_t_pvalue": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_p,
        })

pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df.to_csv("pairwise_model_comparisons.csv", index=False)

print("\n=== PAIRWISE MODEL COMPARISONS (paired bootstrap t-test & Wilcoxon on ECE) ===")
print(pairwise_df.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# CORRELATION: shift magnitude (PSI) vs calibration error (ECE)
# ---------------------------------------------------------------------------
# Pearson tests for a linear relationship; Spearman tests for a monotonic
# relationship without assuming linearity -- reporting both is standard
# practice when the functional form of the relationship isn't known a priori.

print("\n=== CORRELATION: PSI (shift magnitude) vs ECE ===")
correlation_rows = []

# (a) Pooled across all models and shift levels
pearson_r, pearson_p = pearsonr(results_df["PSI"], results_df["ece"])
spearman_r, spearman_p = spearmanr(results_df["PSI"], results_df["ece"])
correlation_rows.append({
    "scope": "All models pooled", "n_points": len(results_df),
    "pearson_r": pearson_r, "pearson_pvalue": pearson_p,
    "spearman_r": spearman_r, "spearman_pvalue": spearman_p,
})

# (b) Per model (n = number of shift levels; small-sample caveat noted for the paper)
for model_name in MODELS:
    sub = results_df[results_df["model"] == model_name]
    pr, pp = pearsonr(sub["PSI"], sub["ece"])
    sr, sp = spearmanr(sub["PSI"], sub["ece"])
    correlation_rows.append({
        "scope": model_name, "n_points": len(sub),
        "pearson_r": pr, "pearson_pvalue": pp,
        "spearman_r": sr, "spearman_pvalue": sp,
    })

correlation_df = pd.DataFrame(correlation_rows)
correlation_df.to_csv("shift_vs_ece_correlation.csv", index=False)
print(correlation_df.round(4).to_string(index=False))

print("\nDone. Files written: results_summary.csv, pairwise_model_comparisons.csv, "
      "shift_vs_ece_correlation.csv, reliability_diagrams.png")