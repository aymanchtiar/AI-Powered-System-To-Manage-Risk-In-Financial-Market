import pandas as pd
import numpy as np
import os


# Load and concatenate the files into a single DataFrame
base_path = '/Users/mehdiamrani/Desktop/FYP_project/1_data/1_raw_data/stocks_data/'

data_files = {
    'AAPL_1d': base_path + '#AAPL_1d_data.csv',
    'APPL_1m': base_path + '#AAPL_1m.csv',
    'AAPL_5m': base_path + '#AAPL_5m_data.csv',
    'AAPL_15m': base_path + '#AAPL_15m_data.csv',
    'AAPL_30m': base_path + '#AAPL_30m_data.csv',
    'AAPL_60m': base_path + '#AAPL_60m_data.csv',
    'AMZN_1d': base_path + '#AMZN_1d_data.csv',
    'AMZN_1m': base_path + '#AMZN_1m.csv',
    'AMZN_5m': base_path + '#AMZN_5m_data.csv',
    'AMZN_15m': base_path + '#AMZN_15m_data.csv',
    'AMZN_30m': base_path + '#AMZN_30m_data.csv',
    'AMZN_60m': base_path + '#AMZN_60m_data.csv',
    'GOOG_1d': base_path + '#GOOG_1d_data.csv',
    'GOOG_1m': base_path + '#GOOG_1m.csv',
    'GOOG_5m': base_path + '#GOOG_5m_data.csv',
    'GOOG_15m': base_path + '#GOOG_15m_data.csv',
    'GOOG_30m': base_path + '#GOOG_30m_data.csv',
    'GOOG_60m': base_path + '#GOOG_60m_data.csv',
}


output_directory = '/Users/mehdiamrani/Desktop/FYP_project/1_data/2_processed_data/stocks'
os.makedirs(output_directory, exist_ok=True)


def process_file(file_path, output_file_name):
    df = pd.read_csv(file_path)
    # lag features and moving averages
    for price_type in ['close']:
        for lag in [1, 3, 5]:
            df[f'{price_type}_lag_{lag}'] = df[price_type].shift(lag)

    # Price changes (absolute and percentage)
    df['close_change_abs'] = df['close'].diff()
    df['close_change_pct'] = df['close'].pct_change()

    # High-Low range
    df['high_low_range'] = df['high'] - df['low']

    # Average price
    df['average_price'] = (df['high'] + df['low'] + df['close']) / 3

    # Volume changes
    df['volume_change'] = df['Volume'].diff()

    # Volume oscillator
    df['volume_oscillator'] = df['Volume'] - df['Average Volume']

    # Moving Averages: Simple Moving Average (SMA), Exponential Moving Average (EMA)
    for window in [5, 10]:
        df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

    # Bollinger Bands
    window = 20
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    df[f'BollingerB_upper_{window}'] = sma + (std * 2)
    df[f'BollingerB_lower_{window}'] = sma - (std * 2)

    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(14).mean()
    roll_down = down.abs().rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14, center=False).min()
    high_max = df['high'].rolling(window=14, center=False).max()
    df['Stochastic_oscillator'] = 100 * ((df['close'] - low_min) / (high_max - low_min))

    # Apply labels
    for i in range(1, 11):
        df[f'Future_Close'] = df['close'].shift(-i)
        df[f'Label_{i}'] = np.where(df[f'Future_Close'] > df['close'], 1, 0)

    # Fill missing values introduced by lags and shifts
    df.fillna(0, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop specified columns from the DataFrame
    columns_to_drop = ['Future_Close', 'volume', 'open', 'high', 'low','Adj Close', 'Datetime','volume_oscillator','volume_change',	'Volume'	,'Average Volume']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Drop the first 20 rows to account for the window period
    df = df.iloc[20:].reset_index(drop=True)

    df.to_csv(f'{output_directory}/{output_file_name}_processed.csv', index=False)

# Process each file
for name, path in data_files.items():
    process_file(path, name)

print("All files processed and saved in '1_data/2_processed_data/stocks' folder.")
