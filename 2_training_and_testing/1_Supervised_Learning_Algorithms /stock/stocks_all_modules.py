import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib


file_info = [
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AAPL_1d_processed.csv", "AAPL", "1d"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AAPL_1m_processed.csv", "AAPL", "1m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AAPL_5m_processed.csv", "AAPL", "5m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AAPL_15m_processed.csv", "AAPL", "15m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AAPL_30m_processed.csv", "AAPL", "30m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AAPL_60m_processed.csv", "AAPL", "60m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AMZN_1d_processed.csv", "AMZN", "1d"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AMZN_1m_processed.csv", "AMZN", "1m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AMZN_5m_processed.csv", "AMZN", "5m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AMZN_15m_processed.csv", "AMZN", "15m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AMZN_30m_processed.csv", "AMZN", "30m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/AMZN_60m_processed.csv", "AMZN", "60m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/GOOG_1d_processed.csv", "GOOG", "1d"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/GOOG_1m_processed.csv", "GOOG", "1m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/GOOG_5m_processed.csv", "GOOG", "5m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/GOOG_15m_processed.csv", "GOOG", "15m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/GOOG_30m_processed.csv", "GOOG", "30m"),
    ("/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks/GOOG_60m_processed.csv", "GOOG", "60m"),
]


# Results DataFrame initialization
results_df = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1", "Lag"])
for file_path, asset, timeframe in file_info:
    df = pd.read_csv(file_path)
    print(f"Processing file: {file_path.split('/')[-1]} for asset: {asset} and timeframe: {timeframe}")
    # Loop through each label for separate training and testing
    for i in range(1, 11):
        label = f'Label_{i}'
        print(f"Processing {label}...")
        labels_to_drop = [f'Label_{j}' for j in range(1, 11) if j != i]
        df_numeric = df.drop(labels_to_drop, axis=1)
        X = df_numeric.drop([label], axis=1)
        y = df[label]

        # Normalizing features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Manual split
        split_idx = int(len(X_scaled) * 0.9)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:-10]
        y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:-10].values

        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=10),
            "MLP": MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='adam', max_iter=300)
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
                "Lag": i ,
                "Asset": asset,
                "Timeframe": timeframe,
            }, ignore_index=True)

results_file_path = "/Users/mehdiamrani/Desktop/FYP_project/2_training_and_testing/4_results/1_Supervised_Learning_Algorithms/all_modules_results_stocks.csv"
results_df.to_csv(results_file_path, index=False)
print(f"Results saved to {results_file_path}.")
