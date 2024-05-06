import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

file_info = [
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/curency_pairs/GBPUSD_1d_processed.csv", "GBPUSD", "1d"),
]

results_df = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1", "Lag", "Asset", "Timeframe"])

best_accuracy = 0
best_model = None
best_model_details = {}
best_scaler = None

for file_path, asset, timeframe in file_info:
    df = pd.read_csv(file_path)
    print(f"Processing file: {file_path.split('/')[-1]} for asset: {asset} and timeframe: {timeframe}")

    for i in range(1, 11):
        label = f'Label_{i}'
        print(f"Processing {label}...")

        labels_to_drop = [f'Label_{j}' for j in range(1, 11) if j != i]
        df_numeric = df.drop(labels_to_drop, axis=1)
        X = df_numeric.drop([label], axis=1)
        y = df[label]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        split_idx = int(len(X_scaled) * 0.9)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:-10]
        y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:-10].values

        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=200),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "SVM": SVC()
        }

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')

            results_df = results_df._append({
                "Classifier": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Lag": i,
                "Asset": asset,
                "Timeframe": timeframe
            }, ignore_index=True)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = clf
                best_scaler = scaler
                best_model_details = {"Classifier": name, "Lag": i, "Asset": asset, "Timeframe": timeframe}

if best_model:
    best_model_path = f"/Users/mehdiamrani/Desktop/FYP_project/3_social_media_posting_algorithm/best_module/best_model_{best_model_details['Asset']}_{best_model_details['Timeframe']}_Lag{best_model_details['Lag']}.joblib"
    best_scaler_path = f"/Users/mehdiamrani/Desktop/FYP_project/3_social_media_posting_algorithm/best_module/best_scaler_{best_model_details['Asset']}_{best_model_details['Timeframe']}_Lag{best_model_details['Lag']}.joblib"
    joblib.dump(best_model, best_model_path)
    joblib.dump(best_scaler, best_scaler_path)
    print(f"Best model and scaler saved to {best_model_path} and {best_scaler_path} with accuracy: {best_accuracy}")

results_file_path = "/Users/mehdiamrani/Desktop/FYP_project/3_social_media_posting_algorithm/best_module/results_of_the_best_module.csv"
results_df.to_csv(results_file_path, index=False)
print(f"Results saved to {results_file_path}.")