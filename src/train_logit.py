from sklearn.linear_model import LogisticRegression
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import shap
from sklearn.metrics import average_precision_score, f1_score
import numpy as np
import pickle

if __name__ == '__main__':

    def objective(trial):
        
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:
            l1_ratio = None

        if penalty == 'elasticnet':
            solver = 'saga'
        elif penalty == 'l1':
            solver = 'saga'
        else:
            solver = 'lbfgs'

        C = trial.suggest_float("C", 1e-3, 1e2, log=True)

        scores = []
        for train_idx, valid_idx in list(tscv.split(X_ts))[:-1]:
            X_train, X_valid = X_ts.iloc[train_idx], X_ts.iloc[valid_idx]
            y_train, y_valid = y_ts.iloc[train_idx], y_ts.iloc[valid_idx]

            model = LogisticRegression(
                penalty = penalty,
                C=C,
                l1_ratio=l1_ratio,
                solver=solver,
                class_weight='balanced',
                max_iter=1000,
                random_state=0,
                #scoring='average_precision'
            )
            model.fit(X_train, y_train)
            y_hat = model.predict_proba(X_valid)[:,1]
            scores.append(average_precision_score(y_valid, y_hat))
        
        return np.mean(scores)

    df = pd.read_csv("data/creditcard.csv")

    tscv = TimeSeriesSplit(n_splits=10)
    df_ts = df.sort_values(by='Time').reset_index(drop=True)
    X_ts, y_ts = df_ts.drop(columns=['Class', 'Time']), df_ts['Class']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best = study.best_params
    logistic_model = LogisticRegression(
        penalty = best['penalty'],
        C=best['C'],
        l1_ratio = best.get('l1_ratio'),
        solver='saga' if best['penalty'] in ('l1', 'elasticnet') else 'lbfgs',
        class_weight = 'balanced',
        max_iter=1000,
        random_state=0
    )
    
    print(f"\nBest AUC-PR: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    train_idx, valid_idx = list(tscv.split(X_ts))[-2]
    _, test_idx = list(tscv.split(X_ts))[-1]
    X_train, y_train = X_ts.iloc[train_idx], y_ts.iloc[train_idx] 
    X_valid, y_valid = X_ts.iloc[valid_idx], y_ts.iloc[valid_idx]
    X_test, y_test = X_ts.iloc[test_idx], y_ts.iloc[test_idx]
    
    logistic_model.fit(X_train, y_train)

    with open('models/logistic.pkl', 'wb') as file:
        pickle.dump(logistic_model, file)

