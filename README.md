# 🚜 Heavy Equipment Selling Price Prediction

A machine learning project that predicts the selling price of used heavy equipment based on historical transaction records, equipment specifications, usage information, and transaction details.

The project focuses on building an accurate regression pipeline for predicting `TargetValue`, the selling price of equipment in USD. Since the competition is evaluated using **Root Mean Squared Logarithmic Error (RMSLE)**, the modelling process works primarily in log-price space.

## 📌 Project Overview

Used heavy equipment such as excavators, wheel loaders, and other industrial machines can have very different resale values depending on their age, operating hours, specifications, utilization, location, and configuration.

The objective of this project is to use historical equipment transactions from `train.csv` to predict the unknown selling prices in `test.csv`.

### Target Variable

`TargetValue` — the observed selling price of the equipment in USD.

### Evaluation Metric

**RMSLE (Root Mean Squared Logarithmic Error)**

RMSLE is suitable for this problem because equipment prices are right-skewed and relative prediction error is more important than absolute dollar error.

---

## 📊 Dataset

The notebook uses four provided files:

* `train.csv` — historical transactions with known `TargetValue`
* `test.csv` — equipment transactions requiring price predictions
* `sample_submission.csv` — required submission format
* `metadata.csv` — descriptions of the dataset features

### Dataset Size

| Dataset  |    Rows | Columns |
| -------- | ------: | ------: |
| Training | 138,701 |      50 |
| Test     |  15,000 |      49 |

The dataset contains numerical, categorical, identification, transaction-date, and technical specification features.

Important variables include:

* `ManufactureYear`
* `OperationalHoursMeter`
* `UtilizationTier`
* `TransactionDate`
* `Spec_FullDescriptor`
* `Spec_BaseClass`
* `Spec_SubClass`
* `FunctionalClassification`
* `RegionCode`
* `InventoryGroupCategory`
* `CabinType`
* `DrivetrainType`
* `TargetValue`

---

## 🔍 Exploratory Data Analysis

The analysis investigates:

* Missing-value patterns
* Duplicate records
* Numerical and categorical distributions
* Feature cardinality
* Target-value distribution
* Relationships between equipment characteristics and selling price
* Correlations and equipment age
* High-cardinality identifiers and technical specification fields

A major observation is that many `colN` specification fields contain substantial missing values. In many cases, this represents **equipment-specific structural missingness** rather than simply unknown data.

The target variable is also strongly right-skewed, making log transformation useful for modelling.

---

## 🧹 Data Preprocessing & Feature Engineering

The modelling pipeline includes:

* Missing-value handling
* Structural missingness indicators
* Equipment age derived from transaction and manufacturing information
* Frequency encoding for high-cardinality features
* K-fold target encoding for selected categorical features
* Ordinal encoding for categorical features used by tree models
* Native categorical handling for CatBoost
* Explicit handling of the `1001` unknown manufacturing-year value
* Training on `log1p(TargetValue)` to align with RMSLE

High-cardinality identifiers such as `AssetID` and `ProductConfigID` are frequency-encoded rather than being used directly as raw numerical magnitudes.

---

## 🤖 Machine Learning Models

The final modelling stage uses **5-fold cross-validation** with three regression models:

1. **XGBoost Regressor**
2. **CatBoost Regressor**
3. **HistGradientBoostingRegressor**

Each model is trained across five folds, and test predictions are averaged across the folds.

The final predictions are combined using **Non-Negative Least Squares (NNLS)** blending in log space.

### Final Blend Weights

| Model                | Weight |
| -------------------- | -----: |
| XGBoost              |   0.32 |
| CatBoost             |   0.68 |
| HistGradientBoosting |   0.00 |

---

## 📈 Model Performance

### 5-Fold Out-of-Fold Performance

| Model                |   OOF RMSLE |
| -------------------- | ----------: |
| XGBoost              |     0.21125 |
| CatBoost             | **0.20422** |
| HistGradientBoosting |     0.23142 |
| **NNLS Ensemble**    | **0.20217** |

The **CatBoost model** achieved the best individual OOF RMSLE, while combining XGBoost and CatBoost through NNLS produced the strongest overall result.

### 🏆 Final Model

**5-Fold XGBoost + CatBoost + HistGradientBoosting NNLS Ensemble**

**Estimated OOF RMSLE: `0.20217`**

The final ensemble is trained using all available labelled data through the 5-fold bagging approach and is then used to generate predictions for the test dataset.

---

## 🔮 Prediction Output

The final predictions are saved in:

```text
submission.csv
```

The file contains:

| Column          | Description                      |
| --------------- | -------------------------------- |
| `TransactionID` | Equipment transaction identifier |
| `TargetValue`   | Predicted selling price          |

Example predictions from the final submission:

```text
TransactionID    TargetValue
1139307          70384.557915
1139419          78596.095997
1139482          30411.279884
1139522          19825.029850
1139684          12240.216842
```

The generated submission contains **15,000 predictions** and follows the format required by `sample_submission.csv`.

---

## 📁 Project Structure

```text
Heavy-Equipment-Selling-Price-Prediction/
│
├── notebooks/
│   └── 23f2003210-notebook-2026t2.ipynb
│
├── project-report/
│   └── ...
│
├── data/
│   └── ...
│
├── README.md
└── submission.csv
```

---

## ⚙️ Requirements

The notebook uses Python and the following major libraries:

```text
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
```

Install the required packages using:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

---

## 🚀 How to Run

1. Clone the repository.

```bash
git clone https://github.com/bhushan-ssh/heavy-equipment-selling-price-prediction.git
cd heavy-equipment-selling-price-prediction
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```text
notebooks/23f2003210-notebook-2026t2.ipynb
```

4. Run the notebook from start to finish.

The notebook performs data exploration, preprocessing, feature engineering, model training, 5-fold cross-validation, ensemble blending, and final test-set prediction.

---

## 📌 Key Findings

* Equipment selling price is strongly right-skewed, making log-space modelling appropriate.
* Equipment age and operational usage provide important information for estimating resale value.
* The dataset contains substantial structural missingness in technical specification fields.
* High-cardinality identifiers require careful encoding to avoid treating IDs as meaningful continuous quantities.
* CatBoost provides the strongest individual OOF performance among the final models.
* The NNLS ensemble improves upon the individual models, achieving an estimated **0.20217 RMSLE**.

---

## 👨‍💻 Author

**Bhushan Sonawane**


