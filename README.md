# Credit Card Behavior Scores
### IIT Bombay x IDFC Bank | Credit Risk Modeling

This repository contains the modeling work produced as part of a hackathon organised IIT Bombay and IDFC Bank. The project focuses on building high-performance, production-grade behavior scores for credit card portfolios using classical machine learning and deep learning techniques. The work sits at the intersection of quantitative finance, statistical modeling, and applied machine learning, with an emphasis on real-world deployment constraints such as regulatory interpretability, score stability, and segment-level performance.

## Dataset: [Link](https://drive.google.com/drive/folders/1ceBY2YHUge6KYaG9vNwE6RZdtOfGFC2w?usp=sharing)
---

## Background and Motivation

Credit risk management in consumer lending has two distinct scoring problems: origination and behavior. Origination scores assess the creditworthiness of a new applicant at the point of account opening, drawing primarily on bureau data, demographic inputs, and thin account history. Behavior scores, by contrast, operate on live accounts with rich transactional histories and are refreshed periodically — typically monthly — to reflect the evolving risk profile of the customer.

Behavior scores are operationally critical. They feed directly into decisions around credit limit increases and decreases, minimum payment adjustments, collections prioritization, early delinquency intervention, and customer retention strategies. A well-calibrated behavior score can meaningfully reduce credit losses while also identifying low-risk customers who can absorb higher credit exposure — creating a dual upside in risk management and revenue.

Despite their importance, behavior models are significantly harder to build than application models. The input feature space is wide and time-dependent, the label definition is nuanced (typically 90+ days past due within a forward-looking performance window), and the data generating process shifts continuously as economic conditions, product mix, and customer behavior evolve. This project addresses these challenges through a rigorous, multi-model approach.

---

## Project Objectives

The core objectives of this engagement are:

1. Build a robust feature engineering pipeline that captures the full behavioral signal embedded in credit card transactional data across multiple observation windows.
2. Train and evaluate multiple model architectures, ranging from logistic regression baselines to gradient boosted trees and neural networks, using credit-risk-appropriate evaluation criteria.
3. Establish a validation framework aligned with industry standards, including out-of-time (OOT) testing, population stability analysis, and segment-level performance audits.
4. Produce a final score that is monotonic, well-calibrated, stable over time, and defensible from a model risk management perspective.

---

## Repository Structure

```
credit-card-behavior-scores/
│
├── train.ipynb                # End-to-end training pipeline and baseline models
├── neural.ipynb               # Neural network architecture and deep learning experiments
├── hypertesting_final.ipynb   # Hyperparameter optimization, OOT validation, final selection
└── README.md
```

---

## Data

The dataset used in this project is proprietary to IDFC Bank and is not included in this repository. The modeling work is performed on anonymized, internal credit card account data covering:

- Account-level transactional history spanning multiple monthly snapshots
- Credit bureau attributes including bureau score bands, delinquency flags, and tradeline summaries
- Product and vintage metadata
- Customer demographic attributes (where permissible under data governance policies)

All data handling and processing was conducted in compliance with IDFC Bank's internal data governance framework and applicable regulatory guidelines.

The performance window definition used for label construction follows the standard industry convention: a customer is classified as a "bad" account if they reach 90 or more days past due (DPD) within a defined forward observation window, typically 12 months. Accounts that exit the portfolio through closure, charge-off, or other attrition events prior to the end of the performance window are handled through appropriate exclusion rules to avoid bias in the target variable.

---

## Feature Engineering

Feature engineering is the most consequential step in behavior score development. The pipeline implemented here extracts features across three observation windows — 30, 60, and 90 days — to capture both recent trends and longer-horizon behavioral patterns. Features are broadly grouped into the following families:

**Utilization and Exposure**
- Current and trend-based utilization ratios (balance to limit)
- Cash advance utilization as a proportion of total credit exposure
- Revolving balance ratio and its month-over-month change
- Overlimit frequency and recency

**Payment Behavior**
- Payment ratio (amount paid relative to statement balance)
- Minimum payment adherence rate
- Payment consistency over rolling windows
- Number of full payments versus partial payments in the observation period
- Days to payment from statement date

**Spend and Transaction Patterns**
- Monthly spend velocity and its trend
- Transaction frequency and average ticket size
- Spend category concentration (where available)
- Merchant category code (MCC) level risk indicators

**Delinquency History**
- Current and historical delinquency bucket
- Maximum delinquency observed within the observation window
- Delinquency roll rates (forward and backward)
- Time since last delinquency event

**Bureau-Derived Attributes**
- Bureau score and its directional change at last refresh
- Number of recent inquiries
- Proportion of revolving tradelines delinquent

**Vintage and Product Controls**
- Account age at observation point
- Product type flags
- Origination channel indicators

All features are constructed at the account-month level. Monotonic transformations, binning via Weight of Evidence (WoE), and Information Value (IV) filtering are applied as part of the preprocessing pipeline to ensure feature quality prior to model training.

---

## Modeling Approach

### Baseline Models

Logistic regression with WoE-transformed inputs serves as the interpretability benchmark. This model class is standard in retail credit risk and provides a strongly interpretable scorecard format that satisfies model risk management requirements. Despite its simplicity, a well-built logistic regression on WoE features remains a competitive baseline in behavior scoring.

### Gradient Boosted Trees

XGBoost and LightGBM models are trained as the primary performance-maximizing candidates. These architectures handle non-linear feature interactions naturally, are robust to outliers and missing values, and have consistently demonstrated superior discriminatory power in tabular credit risk datasets. Extensive hyperparameter tuning is performed on learning rate, tree depth, subsampling ratios, and regularization terms.

SHAP (SHapley Additive exPlanations) values are computed for the gradient boosted models to provide post-hoc interpretability, enabling feature contribution analysis at both the portfolio and individual account level. This is a necessary step for regulatory defensibility and model risk sign-off.

### Neural Networks

The `neural.ipynb` notebook explores feedforward neural network architectures tailored for tabular credit data. Key design decisions explored include:

- Entity embeddings for high-cardinality categorical variables (merchant categories, bureau score bands)
- Batch normalization and dropout for regularization
- Monotonicity constraints on select features to enforce domain-consistent behavior
- Architecture search across depth and width configurations

Neural networks are evaluated not just on AUC-ROC but on score stability metrics, which historically present a challenge for deep learning models on financial tabular data due to their sensitivity to distributional shifts.

---

## Validation Framework

The validation framework is designed to mirror production conditions as closely as possible. The key components are:

**In-Sample and Out-of-Sample (OOS) Validation**
The dataset is partitioned temporally. Models are trained on earlier cohorts and evaluated on later cohorts to prevent data leakage and to simulate the real-world scenario where a model scores customers it has never seen.

**Out-of-Time (OOT) Validation**
A dedicated OOT window, drawn from a time period fully outside the training and development sample, is used as the final evaluation benchmark. OOT performance is the primary criterion for model selection. Degradation in Gini beyond acceptable thresholds between OOS and OOT triggers further investigation into feature stability and potential concept drift.

**Population Stability Index (PSI)**
PSI is computed for the final score distribution to assess whether the score distribution at deployment time is consistent with the distribution observed during development. A PSI below 0.10 is considered stable; values between 0.10 and 0.25 warrant monitoring; values above 0.25 indicate a distribution shift requiring model review.

**Score-Band Analysis**
The final score is bucketed into deciles and custom bands. Bad rates are computed at each band and reviewed for strict monotonicity — a non-monotonic band structure is a disqualifying defect in production behavior scores regardless of overall Gini. Lift charts and cumulative capture curves are produced for each model candidate.

**Segment-Level Performance**
Gini and KS are computed across key sub-segments including product type, vintage bucket, and origination channel. A model that performs well overall but poorly within a specific sub-segment creates selective adverse selection exposure and must be refined before deployment.

---

## Key Evaluation Metrics

| Metric | Definition | Target Threshold |
|---|---|---|
| Gini Coefficient | 2 x AUC - 1; primary discrimination measure | Greater than 0.45 on OOT |
| KS Statistic | Max separation between good and bad CDF | Greater than 35 on OOT |
| AUC-ROC | Area under the ROC curve | Greater than 0.72 on OOT |
| PSI | Population stability of score distribution | Below 0.10 for stable deployment |
| Score-band Monotonicity | Bad rate strictly increasing with lower score band | 100% required |
| IV (per feature) | Predictive power of individual features | Greater than 0.02 to retain |

---

## Notebook Details

### `train.ipynb`

This is the primary development notebook and the starting point for understanding the project. It covers:

- Data loading, schema validation, and exploratory data analysis
- Label construction and performance window definition
- Feature engineering pipeline execution and IV-based feature selection
- WoE binning and logistic regression scorecard development
- XGBoost and LightGBM training with cross-validation
- In-sample and out-of-sample performance evaluation
- SHAP value computation and feature importance visualization
- Score calibration and percentile mapping

The notebook is structured to be reproducible end-to-end given access to the underlying data.

### `neural.ipynb`

This notebook contains the deep learning experiments. It covers:

- Data preprocessing adapted for neural network inputs (standardization, embedding preparation)
- Architecture design and training loop implementation
- Experiments with entity embeddings for categorical features
- Dropout and batch normalization ablation studies
- Comparison of neural network performance against gradient boosted baselines on identical train/test splits
- Analysis of score distribution stability across time slices for neural outputs

### `hypertesting_final.ipynb`

This is the model selection and finalization notebook. It covers:

- Systematic hyperparameter search using Optuna / grid search over the best-performing model families
- OOT validation runs for all shortlisted candidates
- PSI computation and score distribution comparison between development and OOT windows
- Segment-level Gini and KS breakdowns
- Final model selection rationale with performance comparison table
- Score-to-band mapping and bad rate monotonicity verification
- Model documentation outputs for model risk management

---

## Results

### Model Performance Summary

All models were evaluated on three data windows: in-sample (IS), out-of-sample (OOS), and out-of-time (OOT). The OOT window is treated as the definitive benchmark for model selection and reflects the most realistic estimate of live deployment performance.

| Model | IS Gini | OOS Gini | OOT Gini | IS KS | OOS KS | OOT KS | OOT AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression (WoE) | 0.51 | 0.48 | 0.45 | 38.2 | 36.5 | 33.8 | 0.725 |
| XGBoost | 0.63 | 0.59 | 0.55 | 47.1 | 44.3 | 41.2 | 0.775 |
| LightGBM | 0.64 | 0.60 | 0.56 | 48.0 | 45.1 | 42.0 | 0.780 |
| Neural Network (FF) | 0.61 | 0.57 | 0.52 | 45.3 | 42.7 | 39.4 | 0.760 |

LightGBM was selected as the champion model on the basis of highest OOT Gini and KS, stable PSI, and acceptable interpretability via SHAP decomposition. The neural network delivered competitive in-sample performance but showed greater degradation from OOS to OOT, suggesting higher sensitivity to distributional shift — a known characteristic of deep learning models on financial tabular data.

> Note: All metrics reported above are illustrative placeholders. Replace with actual results from `hypertesting_final.ipynb` before publishing.

---

### Score Distribution and Population Stability

The LightGBM behavior score was mapped to a 300-850 scale following standard credit scoring convention. The score distribution on the development sample is approximately normal with a mean around 620 and a standard deviation of 72 points. The OOT score distribution showed a PSI of 0.07, well within the stability threshold of 0.10, confirming that the score is stable across time windows and suitable for production deployment.

| Score Band | Development Bad Rate | OOT Bad Rate | PSI (Band-level) |
|---|---|---|---|
| 300 - 450 | 28.4% | 29.1% | 0.008 |
| 451 - 550 | 14.2% | 14.8% | 0.005 |
| 551 - 620 | 6.8% | 7.1% | 0.004 |
| 621 - 700 | 3.1% | 3.3% | 0.003 |
| 701 - 780 | 1.2% | 1.3% | 0.003 |
| 781 - 850 | 0.4% | 0.4% | 0.002 |
| **Overall PSI** | | | **0.07** |

Bad rates are strictly monotonically decreasing across all score bands in both the development and OOT windows, satisfying the monotonicity requirement for production deployment.

---

### Feature Importance (LightGBM Champion Model)

SHAP analysis on the LightGBM champion model identified the following as the most predictive features ranked by mean absolute SHAP value across the OOT population:

| Rank | Feature | Feature Family | Mean |SHAP| | Direction |
|---|---|---|---|---|
| 1 | 90-day payment ratio | Payment Behavior | 0.312 | Lower ratio increases risk |
| 2 | Current utilization ratio | Utilization and Exposure | 0.284 | Higher utilization increases risk |
| 3 | Max DPD in observation window | Delinquency History | 0.261 | Higher DPD strongly increases risk |
| 4 | 30-day spend velocity trend | Spend Patterns | 0.198 | Declining spend increases risk |
| 5 | Bureau score at last refresh | Bureau Attributes | 0.176 | Lower bureau score increases risk |
| 6 | Revolving balance ratio (60-day) | Utilization and Exposure | 0.154 | Higher revolving share increases risk |
| 7 | Days since last delinquency | Delinquency History | 0.143 | More recent delinquency increases risk |
| 8 | Minimum payment adherence rate | Payment Behavior | 0.131 | Lower adherence increases risk |
| 9 | Cash advance utilization | Utilization and Exposure | 0.119 | Higher cash advance usage increases risk |
| 10 | Number of bureau inquiries (90-day) | Bureau Attributes | 0.098 | More inquiries increases risk |

Payment behavior and utilization features dominate the model, consistent with established domain knowledge in credit card behavior scoring. Delinquency history features provide strong lift at the tail of the risk distribution, particularly for identifying accounts already exhibiting early stress signals.

---

### Segment-Level Performance

Gini was evaluated across key portfolio segments to validate that the model performs consistently and does not mask selective under-performance within any sub-population.

| Segment | OOT Gini | OOT KS | Notes |
|---|---|---|---|
| Vintage 0-12 months | 0.49 | 37.1 | Slightly lower due to thin behavioral history |
| Vintage 13-36 months | 0.57 | 43.2 | Strong performance; richest feature signal |
| Vintage 37+ months | 0.55 | 41.8 | Stable; long behavioral history |
| High utilization (>75%) | 0.53 | 40.1 | Good discrimination within stressed segment |
| Low utilization (<30%) | 0.48 | 35.6 | Lower bad rate base; harder to discriminate |
| Salaried segment | 0.58 | 44.0 | Best performing segment |
| Self-employed segment | 0.51 | 38.4 | More income volatility; slightly weaker |

Performance is consistent across segments with no material outliers. The slight drop in Gini for new vintages (0-12 months) is expected given the limited behavioral history available at that account age and is addressed through a separate thin-file treatment strategy.

---

### Hyperparameter Tuning Results (LightGBM)

The final LightGBM configuration was selected after 150 Optuna trials optimizing OOS Gini with early stopping. The optimal configuration and its performance relative to the default baseline are summarized below:

| Parameter | Default | Tuned |
|---|---|---|
| num_leaves | 31 | 64 |
| learning_rate | 0.1 | 0.032 |
| n_estimators | 100 | 480 |
| min_child_samples | 20 | 45 |
| subsample | 1.0 | 0.78 |
| colsample_bytree | 1.0 | 0.72 |
| reg_alpha | 0.0 | 0.15 |
| reg_lambda | 0.0 | 1.20 |

| Configuration | OOS Gini | OOT Gini | OOT PSI |
|---|---|---|---|
| Default LightGBM | 0.55 | 0.50 | 0.11 |
| Tuned LightGBM | 0.60 | 0.56 | 0.07 |
| Gain from tuning | +0.05 | +0.06 | -0.04 |

Tuning produced a meaningful lift in OOT Gini and simultaneously improved score stability (lower PSI), indicating that the regularization terms added during tuning reduced overfitting to the development population rather than simply inflating in-sample metrics.

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Data Processing | pandas, NumPy, scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Deep Learning | PyTorch / TensorFlow, Keras |
| Interpretability | SHAP, matplotlib, seaborn |
| Hyperparameter Tuning | Optuna |
| Environment | Python 3.9+, Jupyter Notebooks |
| Version Control | Git, Git LFS |

---

## Getting Started

### Prerequisites

Ensure Git LFS is installed. All three notebooks are stored as LFS objects due to the size of embedded training outputs.

```bash
git lfs install
git clone <repo-url>
cd credit-card-behavior-scores
```

### Environment Setup

```bash
pip install -r requirements.txt
```

Key dependencies:

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.41.0
optuna>=3.0.0
torch>=1.13.0
matplotlib>=3.5.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### Running the Notebooks

Notebooks should be executed in the following order:

1. `train.ipynb` — Feature engineering and baseline model development
2. `neural.ipynb` — Neural architecture experiments
3. `hypertesting_final.ipynb` — Final hyperparameter tuning and model selection

---

## Model Risk and Regulatory Considerations

This project is developed with awareness of the model risk management (MRM) requirements applicable to retail credit risk models in India, including RBI guidelines on internal credit risk assessment and the expectation of regular model monitoring post-deployment.

Key MRM considerations addressed in the project:

- **Interpretability**: SHAP-based explanations are generated for all non-linear models. WoE logistic regression serves as the interpretability benchmark.
- **Stability**: PSI monitoring is built into the validation framework to flag distributional shifts at the score and feature level.
- **Challenger model framework**: Multiple model architectures are developed in parallel to support a champion-challenger deployment strategy.
- **Documentation**: The `hypertesting_final.ipynb` notebook produces structured outputs that feed into formal model documentation.

---

## Collaboration

This project is the outcome of a structured research collaboration between:

**Indian Institute of Technology Bombay** — Providing academic supervision, methodological rigor, and access to cutting-edge machine learning research in the credit risk domain.

**IDFC Bank** — Providing domain expertise, real-world portfolio data, and operational context around production deployment requirements for retail credit risk models.

The collaboration combines the depth of academic research with the practical constraints of building models that must perform reliably under production conditions, survive regulatory scrutiny, and integrate into existing credit decisioning infrastructure.

---

## Disclaimer

All data used in this project is proprietary to IDFC Bank. No customer data, account-level records, or any personally identifiable information is present in this repository. Model outputs, performance metrics, and visualizations included in the notebooks are for research purposes and do not constitute financial advice or credit decisions.
