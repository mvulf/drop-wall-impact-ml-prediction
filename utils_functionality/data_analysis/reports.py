from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_curve, accuracy_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import shap
import pandas as pd


def get_classification_report(y_train, y_test, y_preds, y_preds_proba):
    from IPython.display import display, HTML
    display(HTML(f'<h2>ROC AUC: {roc_auc_score(y_test, y_preds):.4f}</h2>'))
    crep = pd.DataFrame(classification_report(
        y_test, y_preds, output_dict=True)).T
    display(HTML(crep.to_html()))


def get_shap_interpretation(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
    plt.show()


def get_cv_results(model, X_train, y_train, X_test, y_test, RANDOM_STATE=42):
    kfold = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    train_results = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring="f1")
    print("F1 Train:\t%.2f%% (std: %.2f%%)" %
          (train_results.mean()*100, train_results.std()*100))

    test_results = cross_val_score(
        model, X_test, y_test, cv=kfold, scoring="f1")
    print("F1 Test:\t%.2f%% (std: %.2f%%)" %
          (test_results.mean()*100, test_results.std()*100))
