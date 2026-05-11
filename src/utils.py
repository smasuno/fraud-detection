import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import shap
import seaborn as sns
import warnings
import pandas as pd
import pickle

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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    eda_df['rolling_pct'].plot(ax=ax[0,0])
    ax[0,0].set_title('Percentage of Transactions Flagged as Fraudulent (Rolling Window)', size=12)
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('Rolling Percentage') # An intermittent pattern, but there are spikes (e.g. at around 7500 mark) where it becomes almost 40% of the transactions

    # T-SNE Cluster Diagram
    df_ts = df.sort_values(by='Time').reset_index(drop=True)
    X_ts, y_ts = df_ts.drop(columns=['Class', 'Time']), df_ts['Class']
    tsne = TSNE(n_components=2, init='pca', random_state=12345, perplexity=50)
    X_tsne = tsne.fit_transform(X_ts[:5000])

    #ax = plt.subplot(aspect='equal')
    colors = y_ts[:5000]
    ax[0,1].scatter(X_tsne[colors==0, 0], X_tsne[colors==0, 1], label='Legitimate', alpha=0.5, color='black')
    ax[0,1].scatter(X_tsne[colors==1, 0], X_tsne[colors==1, 1], label='Fraud', color='red')
    ax[0,1].legend()
    ax[0,1].set_title('T-SNE Cluster Diagram (2-Dimensional)', size=12)

    # Boxplots
    def rescale_func(s: pd.Series):
        u, l = max(s), min(s)
        return 2 * (s - l) / (u - l) - 1
    rescaled_X = df.drop(columns=['Time', 'Class']).apply(func=rescale_func, axis=0)
    sns.boxplot(data=rescaled_X.iloc[:, :10], ax=ax[1,0])
    ax[1,0].set_title('Boxplots of Input Features (Min-Max Rescaled)', size=12)
    
    # Correlation Matrix
    corr_df = X_ts.corr()
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if i >= j:
                corr_df.iloc[i,j] = None
    sns.heatmap(corr_df, cmap = 'viridis', vmin=-1, vmax=1, ax=ax[1,1])
    ax[1,1].set_title("Correlation Heatmap of Input Features")

    plt.tight_layout()
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

def precision_recall_chart(model, y_valid, y_test, X_valid, X_test) -> plt.figure:
    with open('models/logistic.pkl', 'rb') as file:
        logistic_model = pickle.load(file)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    display = PrecisionRecallDisplay.from_estimator(model, X_valid, y_valid, ax=ax[0])
    display.ax_.set_title("Precision-Recall Curve (Validation Set)")
    display = PrecisionRecallDisplay.from_estimator(logistic_model, X_valid, y_valid, ax=ax[0])

    display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax[1])
    display.ax_.set_title("Precision-Recall Curve (Test Set)")
    display = PrecisionRecallDisplay.from_estimator(logistic_model, X_test, y_test, ax=ax[1])

    plt.tight_layout()
    #plt.savefig("precision_recall.png")
    return None

def confusion_matrix_chart(model, y_pred, y_valid, y_test, X_test, best_threshold):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    plot_df = pd.DataFrame({
        'score': y_pred,
        'actual': y_valid,
    })
    plot_df['predicted_class'] = np.where(plot_df['score']>=best_threshold, 1, 0)
    cm = confusion_matrix(plot_df['actual'], plot_df['predicted_class'])
    disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Legitimate', 'Fraud']
        )
    disp.plot(cmap='Blues', colorbar=False, ax=ax[0])
    ax[0].set_title(f"Validation Set (Threshold={best_threshold:.2f})", size=12)

    y_pred_test = model.predict_proba(X_test)[:, 1]
    plot_df = pd.DataFrame({
        'score': y_pred_test,
        'actual': y_test,
    })
    plot_df['predicted_class'] = np.where(plot_df['score']>=best_threshold, 1, 0)
    cm = confusion_matrix(plot_df['actual'], plot_df['predicted_class'])
    disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Legitimate', 'Fraud']
        )
    disp.plot(cmap='Blues', colorbar=False, ax=ax[1])
    ax[1].set_title(f"Test Set (Threshold={best_threshold:.2f})", size=12)
    ax[1].set_ylabel('')
    plt.tight_layout()
    #plt.savefig('confusion_matrix.png', dpi=100)
    return None

def plot_class_distribution(model, y_pred, y_valid, y_test, X_valid, X_test):
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
    #plt.savefig("images/class_distribution_plot.png")
    return None