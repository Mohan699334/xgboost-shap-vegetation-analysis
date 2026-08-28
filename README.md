# Nonlinear climatic thresholds and scale-dependent vegetation responses to climate extremes in an arid–semiarid ecotone of northern China

**Authors:** Lihua Zhang  
**Journal:** Journal of Arid Environments  
**Year:** 2026

---

## Overview

This repository contains the complete R code for reproducing the machine learning analysis presented in the above manuscript. The study uses XGBoost and SHAP to investigate nonlinear responses of vegetation (NDVI) to extreme climate indices in the arid–semiarid ecotone of northern China.

### Key methodological features implemented in this code:

- Annual pixel sampling (10,000 pixels per year)
- 5×5 equal-interval spatial blocking (25 blocks) to avoid spatial autocorrelation
- Hyperparameter grid search with spatial cross-validation
- 5-fold spatial block cross-validation
- Temporal validation (2000–2017 training; 2018–2022 testing)
- SHAP analysis for model interpretability
- LOESS smoothing (span = 0.75) for detecting SHAP zero-crossing turning points
- Equal-interval 95% confidence intervals for SHAP means

---

## Requirements

Install the required R packages before running the script:

install.packages(c("xgboost", "ggplot2", "dplyr", "reshape2"))

---

## Usage

### Option 1: Direct run in RStudio
1. Open XGBoost_SHAP_Analysis.R in RStudio
2. Select all lines (Ctrl+A) and click Run

### Option 2: Command line
Rscript XGBoost_SHAP_Analysis.R

### Data auto-generation
If no data file is found at data/pixel_samples_XGBoost.csv, the script automatically generates a synthetic dataset with known ecological relationships. This allows the code to run immediately for testing and review purposes.

---

## Data format

To run the script with your own data, place a CSV file at:

data/pixel_samples_XGBoost.csv

The CSV must contain the following columns:

| Column | Description |
|--------|-------------|
| year | Calendar year (e.g., 2000–2022) |
| x | Longitude or x-coordinate |
| y | Latitude or y-coordinate |
| NDVI | Target variable (Normalized Difference Vegetation Index) |
| CDD | Consecutive dry days |
| RX5day | Maximum 5-day precipitation |
| TXx | Maximum daily maximum temperature |
| TNn | Minimum daily minimum temperature |
| DTR | Diurnal temperature range |
| TX10p | Cool days (TX < 10th percentile) |
| TN10p | Cool nights (TN < 10th percentile) |
| TX90p | Warm days (TX > 90th percentile) |
| TN90p | Warm nights (TN > 90th percentile) |

Note: All columns are required and must contain numeric values.

---

## Outputs

After successful execution, the results/ folder will contain:

| File | Description |
|------|-------------|
| SHAP_dependence_CDD.png | SHAP dependence plot for CDD |
| SHAP_dependence_RX5day.png | SHAP dependence plot for RX5day |
| SHAP_dependence_TXx.png | SHAP dependence plot for TXx |
| SHAP_dependence_TNn.png | SHAP dependence plot for TNn |
| SHAP_dependence_DTR.png | SHAP dependence plot for DTR |
| SHAP_dependence_TX10p.png | SHAP dependence plot for TX10p |
| SHAP_dependence_TN10p.png | SHAP dependence plot for TN10p |
| SHAP_dependence_TX90p.png | SHAP dependence plot for TX90p |
| SHAP_dependence_TN90p.png | SHAP dependence plot for TN90p |

Each plot displays:
- Points: Individual SHAP values for each sample
- Blue line: LOESS-smoothed trend (span = 0.75)
- Gray ribbon: 95% confidence interval (equal-interval bins)
- Red dashed line: SHAP = 0 reference
- Red dot-dash lines: Detected zero-crossing turning points (where SHAP crosses from negative to positive or vice versa)

---
### Data and Reproducibility

The processed pixel-level dataset generated and compiled for this study is not publicly distributed. Therefore, this repository is intended to provide a transparent and traceable implementation of the analytical workflow rather than a complete dataset that can directly reproduce all numerical results reported in the manuscript. Without access to the original study dataset, users can inspect and run the analytical workflow, but the empirical results reported in the manuscript cannot be fully reproduced from this repository alone.

For users who do not have access to the study dataset, synthetic data may be used to test whether the code runs correctly and to demonstrate the analytical workflow. The synthetic data are provided solely for software testing and methodological demonstration. They do not represent the actual observations used in this study and cannot be used to reproduce the empirical results reported in the manuscript.

The original data sources and their access information are described in the Data Availability statement of the manuscript.

## Contact

For questions about the code or the analysis, please contact the corresponding author:

Lihua Zhang
zhanglihualz@126.com
