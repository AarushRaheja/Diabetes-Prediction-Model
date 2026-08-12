# Diabetes Prediction Model

Binary classifier for early-stage Type 2 diabetes on the [Kaggle diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (100k patient records). I trained a few models, compared them against a majority-class baseline, and pulled out feature importances to see which clinical variables were doing the work.

Methodology from an earlier version of this project was published in RARS and NHSJS.

## Dataset

- 100,000 rows, ~8.5% positive class (imbalanced)
- 8 features: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
- Target: diabetes (0/1)

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Dummy (majority class) | 91.56% | baseline, 0% recall on positives |
| Logistic Regression | 95.29% | default settings |
| Logistic Regression (L1, C=0.1) | 96.03% | best linear model |
| Random Forest | see notebook | ensemble baseline |

Accuracy alone is misleading here because of the class imbalance, so the notebooks also print precision, recall, F1, and a confusion matrix per model.

## Feature importance (L1 logistic)

Top coefficients:

1. heart_disease
2. HbA1c_level
3. gender
4. hypertension
5. bmi

Glucose level also shows up strongly once you scale it.

## Repo layout

```
Poly1 (1).ipynb   baseline + logistic regression + L1
Poly2.ipynb       random forest and comparison
train.py          cleaned-up training script (upcoming)
```

## Run it

```bash
git clone https://github.com/AarushRaheja/Diabetes-Prediction-Model
cd Diabetes-Prediction-Model
pip install -r requirements.txt
jupyter lab
```

Download the dataset from Kaggle and drop `diabetes_prediction_dataset.csv` into the repo root.

## Next

- Add ROC-AUC and PR-AUC as headline metrics instead of accuracy
- Try SMOTE / class weighting to handle the imbalance properly
- Calibration curves for the top model
- Fairness check across gender and age groups
