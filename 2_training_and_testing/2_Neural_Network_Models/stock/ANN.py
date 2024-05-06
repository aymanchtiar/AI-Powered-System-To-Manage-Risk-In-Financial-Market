import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


file_info = [
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AAPL_1d_processed.csv", "AAPL", "1d"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AAPL_1m_processed.csv", "AAPL", "1m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AAPL_5m_processed.csv", "AAPL", "5m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AAPL_15m_processed.csv", "AAPL", "15m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AAPL_30m_processed.csv", "AAPL", "30m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AAPL_60m_processed.csv", "AAPL", "60m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AMZN_1d_processed.csv", "AMZN", "1d"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AMZN_1m_processed.csv", "AMZN", "1m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AMZN_5m_processed.csv", "AMZN", "5m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AMZN_15m_processed.csv", "AMZN", "15m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AMZN_30m_processed.csv", "AMZN", "30m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/AMZN_60m_processed.csv", "AMZN", "60m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/GOOG_1d_processed.csv", "GOOG", "1d"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/GOOG_1m_processed.csv", "GOOG", "1m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/GOOG_5m_processed.csv", "GOOG", "5m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/GOOG_15m_processed.csv", "GOOG", "15m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/GOOG_30m_processed.csv", "GOOG", "30m"),
    ("/content/drive/MyDrive/PROJECT_STOCK/old_data2/GOOG_60m_processed.csv", "GOOG", "60m"),
]


# Function to create ANN model
def create_ann(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Asset", "Timeframe", "Classifier", "Accuracy", "Precision", "Recall", "F1", "Lag"])

scaler = StandardScaler()

for file_path, asset, timeframe in file_info:
    df = pd.read_csv(file_path)
    print(f"Processing file: {file_path} for asset: {asset} and timeframe: {timeframe}")
    
    for i in range(1, 11):  
        label = f'Label_{i}'
        print(f"Processing {label}...")
        
        labels_to_drop = [f'Label_{j}' for j in range(1, 11) if j != i]
        X = df.drop(labels_to_drop + [label], axis=1).select_dtypes(include=[np.number])
        y = df[label]

        # Normalize features before splitting
        X_scaled = scaler.fit_transform(X)

        # Manually splitting the dataset to maintain temporal order
        split_idx = int(len(X_scaled) * 0.9)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:-10]
        y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:-10].values

        # Create and train ANN model
        ann_clf = create_ann(input_dim=X_train.shape[1])
        ann_clf.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict and evaluate
        y_pred = (ann_clf.predict(X_test) > 0.5).astype("int32").flatten()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        # Append the results
        results_df = results_df._append({
            "Asset": asset,
            "Timeframe": timeframe,
            "Classifier": "ANN",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Lag": i,
            "Asset": asset,
            "Timeframe": timeframe
        }, ignore_index=True)

# Save the results to CSV
results_file_path = "/Users/mehdiamrani/Desktop/FYP_project/2_training_and_testing/4_results/2_Neural_Network_Models/ANN_stock.csv"
results_df.to_csv(results_file_path, index=False)
print(f"Results saved to {results_file_path}.")
