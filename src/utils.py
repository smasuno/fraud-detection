import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import shap
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def eda_plot1(df) -> plt.figure:
    eda_df = df.groupby(['Time', 'Class']).apply(lambda x: x.shape[1]).reset_index()
    eda_df.columns = ['Time', 'Class', 'Num']
    eda_df = eda_df.pivot(index='Time', columns='Class', values='Num').fillna(0).reset_index()
    eda_df.columns = ['Time', 'Count_0', 'Count_1']

    eda_df['txn_cnt'] = eda_df['Count_0'] + eda_df['Count_1']
    eda_df['rolling_fraud'] = eda_df.rolling(window=10, on=eda_df.index)['Count_1'].sum()
    eda_df['rolling_sum'] = eda_df.rolling(window=10, on=eda_df.index)['txn_cnt'].sum()
    eda_df['pct'] = eda_df['Count_1'] / eda_df[['Count_0', 'Count_1']].sum(axis=1)
    eda_df['rolling_pct'] = eda_df['rolling_fraud'] / eda_df['rolling_sum']

    fig, ax = plt.subplots(figsize=(10, 6))
    eda_df['rolling_pct'].plot(ax=ax)
    ax.set_title('Percentage of Transactions Flagged as Fraudulent (Rolling Window)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling Percentage') # An intermittent pattern, but there are spikes (e.g. at around 7500 mark) where it becomes almost 40% of the transactions

    return fig

def score_distribution_plot(y_pred, y_valid) -> plt.figure:
    plot_df = pd.DataFrame({
        'score': y_pred,
        'class': y_valid
    })
    plot_df['label'] = np.where(plot_df['class']==1, 'Fraud', 'Legitimate')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=plot_df, x='score', hue='label', stat='probability', common_norm=False, bins=50, alpha=0.5, ax=ax)
    ax.set_xlabel("Predicted Score")
    ax.set_title("Score Distribution by Class (Validation Set)")
    return None

def beeswarm_plot(X_train, model) -> plt.figure:
    #fig, ax = plt.subplots(figsize=(12, 8))
    X100 = shap.utils.sample(X_train, 100)
    explainer_xgb = shap.Explainer(model, X100)
    shap_values_xgb = explainer_xgb(X100)
    shap.plots.beeswarm(shap_values_xgb, show=False)

    plt.title('Feature Importance of XGBoost Model', size=14)
    plt.tight_layout()
    plt.show()
    return None

def precision_recall_chart1(model, X_valid, y_valid) -> plt.figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    display = PrecisionRecallDisplay.from_estimator(model, X_valid, y_valid, ax=ax)
    display.ax_.set_title("Precision-Recall Curve on the Validation Set")
    return None

def confusion_matrix_chart1(plot_df, best_threshold) -> plt.figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df['predicted_class'] = np.where(plot_df['score']>=best_threshold, 1, 0)
    cm = confusion_matrix(plot_df['actual'], plot_df['predicted_class'])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legitimate', 'Fraud']
    )
    disp.plot(cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix at Threshold: {best_threshold:.2f}", size=14)
    return None

def plot_class_distribution(y_pred, y_valid, y_test, X_valid, X_test):
    import seaborn as sns
    import pickle
    import pandas as pd

    with open('models/logistic.pkl', 'rb') as file:
        logistic_model = pickle.load(file)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    plt.suptitle("XGBoost vs Logistic Regression: Score Distribution", size=14)

    plot_df = pd.DataFrame({
        'score': y_pred,
        'class': y_valid
    })
    plot_df['label'] = np.where(plot_df['class']==1, 'Fraud', 'Legitimate')
    #fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=plot_df, x='score', hue='label', stat='probability', common_norm=False, bins=50, alpha=0.5, ax=ax[0,0])
    ax[0,0].set_xlabel(None)
    ax[0,0].set_title("XGBoost - Validation Set")
    #ax[0].legend(loc='upper right')


    plot_df = pd.DataFrame({
        'score': model.predict_proba(X_test)[:,1],
        'class': y_test
    })
    plot_df['label'] = np.where(plot_df['class']==1, 'Fraud', 'Legitimate')
    #fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=plot_df, x='score', hue='label', stat='probability', common_norm=False, bins=50, alpha=0.5, ax=ax[1,0])
    ax[1,0].set_xlabel("Predicted Score")
    ax[1,0].set_title("XGBoost - Test Set")
    #plt.tight_layout()
    #plt.savefig("images/class_distribution.png")

    plot_df = pd.DataFrame({
        'score': logistic_model.predict_proba(X_valid)[:,1],
        'class': y_valid
    })
    plot_df['label'] = np.where(plot_df['class']==1, 'Fraud', 'Legitimate')
    #fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=plot_df, x='score', hue='label', stat='probability', common_norm=False, bins=50, alpha=0.5, ax=ax[0,1])
    ax[0,1].set_xlabel(None)
    ax[0,1].set_title("Logistic Regression - Validation Set")
    #ax[0].legend(loc='upper right')


    plot_df = pd.DataFrame({
        'score': logistic_model.predict_proba(X_test)[:,1],
        'class': y_test
    })
    plot_df['label'] = np.where(plot_df['class']==1, 'Fraud', 'Legitimate')
    #fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=plot_df, x='score', hue='label', stat='probability', common_norm=False, bins=50, alpha=0.5, ax=ax[1,1])
    ax[1,1].set_xlabel("Predicted Score")
    ax[1,1].set_title("Logistic Regression - Test Set")
    plt.tight_layout()
    plt.savefig("images/class_distribution_plot.png")