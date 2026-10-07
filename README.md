# ⚾ Baseball Case Study – Predicting MLB Team Wins

A machine learning project that predicts how many games a Major League Baseball team will win in a season, using its batting, pitching and fielding statistics.

**Final model:** Lasso Regression · **Cross-validated R²:** 0.78 · **Average error:** ~3.7 wins

---

## 📌 Problem Statement

Using statistics from the **2014 MLB season**, build a model that predicts the number of **wins (W)** for a team based on 16 performance indicators. Because the target is a number, this is a **regression** problem.

## 📂 Dataset

- **Source:** [baseball.csv](https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv)
- **Size:** 30 rows (teams) × 17 columns
- **Target:** `W` – Wins
- **No missing values and no duplicate rows**

| Category | Features |
|---|---|
| Batting | `R` Runs · `AB` At bats · `H` Hits · `2B` Doubles · `3B` Triples · `HR` Home runs · `BB` Walks · `SO` Strikeouts · `SB` Stolen bases |
| Pitching | `RA` Runs allowed · `ER` Earned runs · `ERA` Earned run average · `CG` Complete games · `SHO` Shutouts · `SV` Saves |
| Fielding | `E` Errors |

## 🔄 Project Workflow

| Step | What was done | Why |
|---|---|---|
| 1. Data loading | Loaded the CSV with pandas | Starting point |
| 2. Data understanding | Checked shape, types, missing values, duplicates, statistics, skewness | Know the data before changing it |
| 3. EDA | Histograms, boxplots, scatter + trend lines, correlation heatmap | Find patterns, outliers and relationships |
| 4. Data cleaning | Removed outliers (Z-score) and fixed skewness (Yeo-Johnson) | Extreme and lopsided values mislead models |
| 5. Feature selection | Removed multicollinear features using VIF | Duplicate information makes models unstable |
| 6. Split & scale | 80/20 train-test split, StandardScaler fit on train only | Fair evaluation, no data leakage |
| 7. Model building | Trained and compared 10 regression models | No single model is best for every dataset |
| 8. Cross-validation | 5-fold CV with a Pipeline (scaler inside each fold) | A 6-team test set alone is unreliable |
| 9. Hyperparameter tuning | GridSearchCV on Lasso `alpha` | Find the best penalty strength |
| 10. Save & conclude | Saved the model with joblib, predicted wins for new teams | Make the model reusable |

## 🧹 Key Data Preparation Decisions

- **Outliers:** Z-score removed only **1 row (3.3% data loss)**, while IQR would have removed **10 rows (33%)**. Z-score was chosen because the dataset is very small.
- **Skewness:** `CG`, `SHO`, `SV`, `E` were transformed with Yeo-Johnson. `H` was left unchanged because the transform collapsed it to a constant (a numerical precision issue), and it has almost no correlation with wins.
- **Multicollinearity:** `ER`, `ERA` and `RA` were almost identical (VIF up to **2087**). `ER` and `RA` were dropped one at a time, keeping `ERA` (the strongest correlation with wins, −0.83). All remaining VIF values are below 10.

## 🤖 Model Comparison

Ten models were compared using 5-fold cross-validation:

![Model comparison](images/model_comparison.png)

| Model | Test R² | CV R² (mean) | CV R² (std) |
|---|---|---|---|
| **Lasso** | **0.797** | **0.777** | **0.068** |
| Ridge | 0.741 | 0.635 | 0.101 |
| Random Forest | 0.531 | 0.401 | 0.194 |
| XGBoost | 0.633 | 0.399 | 0.270 |
| Linear Regression | 0.783 | 0.392 | 0.239 |
| AdaBoost | 0.429 | 0.366 | 0.174 |
| KNN | 0.433 | 0.364 | 0.213 |
| Gradient Boosting | 0.472 | 0.315 | 0.236 |
| Decision Tree | 0.067 | −0.067 | 0.221 |
| SVR | 0.090 | −0.312 | 0.533 |

**Observations:**
- **Lasso** was the best and most consistent model: its single test score and CV score almost match.
- **Linear Regression's** good test score (0.78) was luck. Its CV score was only 0.39, with fold scores ranging from 0.07 to 0.78.
- **Tree-based models overfit**: they scored R² = 1.0 on training data but much lower on unseen data, because 30 rows is too few for them.
- **Regularisation** (Lasso, Ridge) works best on small datasets because the penalty stops the model from fitting noise.

## 🏆 Final Model

**Lasso Regression** with `alpha = 0.75` (tuned via GridSearchCV)

| Metric | Score |
|---|---|
| Cross-validation R² | 0.783 |
| Test R² | 0.794 |
| Test MAE | 3.70 wins |
| Test RMSE | 4.45 wins |

On average, the model predicts a team's wins to within about **3–4 games**.

## 💡 Key Insights

![Feature importance](images/feature_importance.png)

- **ERA (Earned Run Average) has the biggest impact on winning.** The fewer runs a team's pitchers allow, the more games it wins.
- **Saves (SV)** and **Runs scored (R)** are the next most important factors.
- Lasso automatically reduced **14 features to 5**: ERA, SV, R, SHO and CG.
- **Pitching and defence influence wins more than batting.**

## ⚠️ Limitations

- Very small dataset (30 teams, one season), so results may differ for other seasons.
- The test set has only 6 teams; cross-validation was used to compensate.
- Outlier removal and the power transform were fitted on the full dataset. A complete pipeline would remove this minor data leakage.

## 🚀 Future Improvements

- Use data from multiple seasons to increase the sample size.
- Put every preprocessing step inside a single scikit-learn Pipeline.
- Engineer features such as **Run Differential (R − RA)**, a well-known predictor of wins.

## 📁 Repository Structure

```
Baseball_Project/
├── Baseball_Project.ipynb     # Full analysis notebook (Steps 1–10)
├── baseball_model.pkl         # Saved final model (StandardScaler + Lasso pipeline)
├── power_transformer.pkl      # Saved Yeo-Johnson transformer for CG, SHO, SV, E
├── images/                    # Charts used in this README
├── requirements.txt           # Python libraries needed
└── README.md
```

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Baseball-Wins-Prediction.git
   cd Baseball-Wins-Prediction
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   python -m notebook Baseball_Project.ipynb
   ```
4. Click **Kernel → Restart & Run All**.

### Use the saved model to predict a new team

```python
import joblib
import pandas as pd

model = joblib.load('baseball_model.pkl')
pt = joblib.load('power_transformer.pkl')

team = pd.DataFrame([{'R': 750, 'AB': 5500, 'H': 1420, '2B': 290, '3B': 30, 'HR': 180,
                      'BB': 500, 'SO': 1200, 'SB': 80, 'ERA': 3.40, 'CG': 4,
                      'SHO': 14, 'SV': 50, 'E': 85}])

team[['CG', 'SHO', 'SV', 'E']] = pt.transform(team[['CG', 'SHO', 'SV', 'E']])
print(model.predict(team))   # ≈ 95 wins
```

## 🛠️ Tools & Libraries

Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Statsmodels · Scikit-learn · XGBoost · Joblib · Jupyter Notebook

## 👤 Author

**Nagarjuna Vaddineni**
Internship Project – Phase 1 Evaluation
