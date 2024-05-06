import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# Function to create LSTM model
def create_lstm(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

        # Normalize features
        X_scaled = scaler.fit_transform(X)
        # Reshape for LSTM input
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Split the data
        split_idx = int(len(X_scaled) * 0.9)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:].values

        # Create and train LSTM model
        lstm_model = create_lstm((X_train.shape[1], X_train.shape[2]))
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test, y_test),
                       callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

        # Predict and evaluate
        y_pred = (lstm_model.predict(X_test) > 0.5).astype("int32").flatten()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        # Append the results
        results_df = results_df._append({
            "Asset": asset,
            "Timeframe": timeframe,
            "Classifier": "LSTM",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Lag": i,
            "Asset": asset,
            "Timeframe": timeframe
        }, ignore_index=True)

# Save the results to CSV
results_file_path = "/Users/mehdiamrani/Desktop/FYP_project/2_training_and_testing/4_results/2_Neural_Network_Models/LSTM_stock.csv"
results_df.to_csv(results_file_path, index=False)
print(f"Results saved to {results_file_path}.")
