import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path, label_name):
    df = pd.read_csv(file_path)

    # Exclude all other 'Label_i' columns except the current label
    labels_to_drop = [f'Label_{i}' for i in range(1, 11) if f'Label_{i}' != label_name]
    df = df.drop(labels_to_drop, axis=1)

    # Ensure all remaining columns are numeric
    df_numeric = df.select_dtypes(include=[np.number])
    X = df_numeric.drop([label_name], axis=1)
    y = df[label_name]

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Reshape for CNN input
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    # Manually splitting the dataset to maintain temporal order
    split_idx = int(len(X_scaled) * 0.9)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:].values

    return X_train, X_test, y_train, y_test

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Asset", "Timeframe", "Label", "Accuracy"])

# Updated file_info with the paths, assets, and timeframes
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

for file_path, asset, timeframe in file_info:
    print(f"Processing file: {file_path} for asset: {asset} and timeframe: {timeframe}")
    # Iterate over each label
    for i in range(1, 11):
        label_name = f'Label_{i}'
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, label_name)

        # Build and train the CNN model
        cnn_model = build_cnn((X_train.shape[1], 1))
        history = cnn_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=0)

        # Evaluate the model
        _, accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy for {label_name}: {accuracy}")

        # Append results with asset and timeframe information
        results_df = results_df._append({"Asset": asset, "Timeframe": timeframe, "Label": label_name, "Accuracy": accuracy}, ignore_index=True)

# Save the results to a CSV file
results_file_path = '/Users/mehdiamrani/Desktop/FYP_project/2_training_and_testing/4_results/2_Neural_Network_Models/CNN_stock.csv'
results_df.to_csv(results_file_path, index=False)

print(f"Results saved to {results_file_path}")
