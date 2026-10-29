# Data

## Dataset
**IBM Telco Customer Churn** — a customer-level table from a fictional telecom company, widely used as a churn-prediction benchmark.

| Property | Value |
|---|---|
| File | `telco.csv` (kept in the **repository root**, one level above this folder) |
| Rows | 7,043 customers |
| Columns | 21 (20 features + target) |
| Target | `Churn` (`Yes` / `No`); about 26.5% of customers churned |
| Source | IBM sample data: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv |

The pipeline (`src/calibration_shift_pipeline.py`) downloads the file automatically if `telco.csv` is missing. To fetch it manually, from the repository root:

```bash
curl -L -o telco.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Columns
- **Identifier (dropped):** `customerID`
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account:** `tenure` (months, 0–72), `Contract` (Month-to-month / One year / Two year), `PaperlessBilling`, `PaymentMethod`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Charges:** `MonthlyCharges`, `TotalCharges`
- **Target:** `Churn`

## Cleaning applied in the pipeline
- `customerID` is dropped.
- `TotalCharges` is stored as text; 11 rows (customers with `tenure = 0`) are blank. They are converted to numbers and filled with `0`.
- `Churn` is encoded as `1` for `Yes` and `0` for `No`.
- Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) are standardised and categorical features are one-hot encoded, with the encoders fitted on the base training split only.

## Train / shift splits
Both axes below are defined on `Contract` and `tenure`, and are used only to build the experiment's populations:

| Population | Definition | n |
|---|---|---|
| Base (train, 70% of base set) | Month-to-month, tenure 0–12 months | 1,395 |
| Shift 0 (in-distribution control) | Held-out 30% of the same base set | 599 |
| Shift 1 (mild) | Month-to-month, tenure 13–24 | 737 |
| Shift 2 (moderate) | One year, tenure 25–48 | 518 |
| Shift 3 (strong) | Two year, tenure 49–60 | 277 |
| Shift 4 (severe) | Two year, tenure 61–72 | 986 |

The split uses a fixed random seed (42), so results are reproducible.

## Attribution
The dataset is an IBM sample dataset distributed for learning and demonstration purposes. Please refer to the source repository for its terms of use.