# Loan Approval Prediction Model

This project builds and evaluates multiple classification models to support data-driven loan approval decisions using the LendingClub dataset. The models predict the probability of loan default, enabling financial institutions to assess credit risk and optimize lending policies.

Loan default is a critical risk for lending institutions. This project addresses:
- Identification of high-risk applicants before loan issuance.
- Optimization of approval thresholds based on business constraints (precision vs. recall trade-offs).
- Segmentation of applicants by risk level for differentiated lending strategies.

`loan_approval_modeling.ipynb` contains the entire workflow.

You can also go to the blogpost explaining the project: https://medium.com/@prathik.codes/predicting-loan-default-with-machine-learning-1295afdea799

## Dataset

- **Source**: LendingClub public dataset of historical loan records
- **Target**: `loan_status` → Binary classification (1 = Charged Off/Default, 0 = Fully Paid/Current)
- **Features**: Numeric and categorical attributes (grade, term, purpose, verification status, etc.)

### Data Preprocessing

- **Feature removal**: Eliminated post-loan outcome leakage, identifiers, and text-heavy columns
- **Missing values**: 
  - Numeric features: Median imputation
  - Categorical features: Most frequent imputation
- **Feature scaling**: StandardScaler applied to numeric features
- **Categorical encoding**: One-hot encoding with `handle_unknown='ignore'`
- **Train-test split**: 80-20 stratified split (random seed: 69)

## Models Evaluated

The following classification models were trained and compared:

1. **Logistic Regression**
   - Fast baseline model
   - Interpretable coefficients

2. **Linear SVM (LinearSVC)**
   - Effective for high-dimensional data
   - Class-weighted for imbalanced classes

3. **Random Forest**
   - Ensemble method with 300 trees
   - Feature importance available
   - Balanced subsampling for class imbalance

4. **LightGBM**
   - Gradient boosting framework
   - Fast training with 400 estimators
   - Memory-efficient

5. **XGBoost**
   - Advanced gradient boosting
   - Best overall performance in this study
   - Further tuned with Optuna for hyperparameter optimization
   

## Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted defaults that are actual defaults
- **Recall**: Proportion of actual defaults caught by the model
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **PR-AUC**: Area under the Precision-Recall curve

