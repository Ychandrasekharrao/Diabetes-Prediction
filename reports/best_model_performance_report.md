# Model Performance Report

**Report generated on:** 2025-09-17 12:29:32

**[WINNER] Best Performing Model (based on ROC AUC, then Recall):** `LightGBM`

## Model Comparison Table

```
                 Model  Accuracy  Precision    Recall  F1-Score   ROC AUC
0             LightGBM  0.732612   0.749240  0.691478  0.719201  0.799313
1              XGBoost  0.731807   0.750484  0.686752  0.717205  0.795877
2             AdaBoost  0.724567   0.762445  0.644661  0.698624  0.793580
3  Logistic Regression  0.724932   0.755605  0.657067  0.702899  0.792625
4                  SVM  0.729174   0.754226  0.672131  0.710816  0.778078
5        Random Forest  0.697506   0.694925  0.693694  0.694309  0.755879
6                  KNN  0.704454   0.707132  0.688229  0.697553  0.753444
```

---

## Detailed Analysis of Best Model

### Classification Report

```
              precision    recall  f1-score   support

           0       0.72      0.77      0.74      6902
           1       0.75      0.69      0.72      6771

    accuracy                           0.73     13673
   macro avg       0.73      0.73      0.73     13673
weighted avg       0.73      0.73      0.73     13673

```

### Confusion Matrix

![Confusion Matrix](figures\best_model_analysis\1_confusion_matrix.png)

### ROC AUC Curve

![ROC AUC Curve](figures\best_model_analysis\2_roc_auc_curve.png)

### Predicted Probabilities Distribution

![Predicted Probabilities](figures\best_model_analysis\3_predicted_probabilities.png)

---

## Model Explainability (SHAP)

### SHAP Summary Plot

![SHAP Summary](figures\best_model_analysis\4_shap_summary.png)

### SHAP Dependence Plots

![SHAP Dependence Grid](figures\best_model_analysis\5_shap_dependence_grid.png)

