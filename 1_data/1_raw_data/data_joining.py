import pandas as pd

existing_df = pd.read_csv('/Users/mehdiamrani/Desktop/FYP_project/1_data/1_raw_data/stocks_data/#GOOG_1m_data.csv')

new_data_df = pd.read_csv('/Users/mehdiamrani/Desktop/FYP_project/1_data/1_raw_data/stocks_data/#GOOG_1m_data 2024.csv')

concatenated_df = pd.concat([existing_df, new_data_df], ignore_index=True)

output_path = '/Users/mehdiamrani/Desktop/FYP_project/1_data/1_raw_data/stocks_data/#GOOG_1m.csv'
concatenated_df.to_csv(output_path, index=False)

output_path 
