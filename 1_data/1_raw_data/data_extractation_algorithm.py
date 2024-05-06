import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import pytz


# Initialize MetaTrader 5
if not mt5.initialize():
    print("Initialization failed, error code =", mt5.last_error())
    quit()
else:
    print("Initialization successful")

# Login to MetaTrader 5 , if you using this code you need to create a metatrader 5 demo acount , these crodentionals are expired
login = 42461903
password = 'B#LqH28pxsCq$O'
server = 'AdmiralMarkets-Demo'

if not mt5.login(login, password, server):
    print("Login failed, error code =", mt5.last_error())
    mt5.shutdown()
    quit()
else:
    print("Login successful")



def fetch_and_save_mt5_data(symbol, interval, start_date, end_date, file_name):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    timezone = pytz.timezone("Etc/UTC")
    start_dt = timezone.localize(datetime.strptime(start_date, "%Y-%m-%d"))
    end_dt = timezone.localize(datetime.strptime(end_date, "%Y-%m-%d"))

    # Convert interval to MT5 timeframe
    mt5_timeframes = {
        '1m': mt5.TIMEFRAME_M1,
        '5m': mt5.TIMEFRAME_M5,
        '15m': mt5.TIMEFRAME_M15,
        '30m': mt5.TIMEFRAME_M30,
        '60m': mt5.TIMEFRAME_H1,
        '1d': mt5.TIMEFRAME_D1
    }
    timeframe = mt5_timeframes[interval]

    # Fetch rates
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)

    if rates is None:
        print(f"No data for {symbol}, interval {interval}")
        return

    # Create DataFrame
    ohlc_data = pd.DataFrame(rates)
    ohlc_data['time'] = pd.to_datetime(ohlc_data['time'], unit='s')

    # Rename columns
    ohlc_data.rename(columns={'time': 'Datetime', 'tick_volume': 'Volume'}, inplace=True)

    # Calculate average volume
    average_volume = ohlc_data['Volume'].mean()
    ohlc_data['Average Volume'] = average_volume

    # Select relevant columns
    selected_columns = ['Datetime', 'open', 'high', 'low', 'close', 'Volume', 'Average Volume']
    ohlc_data = ohlc_data[selected_columns]

    # Save to a CSV file
    ohlc_data.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")

    # Shutdown MT5 connection
    mt5.shutdown()

def main():
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY','#AAPL','#AMZN','#GOOG']
    intervals = ['5m','15m','30m','60m','1d']
    start_date =  '2023-01-01'
    end_date = '2023-12-29'

    for symbol in currency_pairs:
        for interval in intervals:
            file_name = f"{symbol}_{interval}data.csv"
            #file_name = f"{symbol}_{interval}_{start_date}_to_{end_date}data.csv"
            fetch_and_save_mt5_data(symbol, interval, start_date, end_date, file_name)

if __name__ == "__main__":
    main()
