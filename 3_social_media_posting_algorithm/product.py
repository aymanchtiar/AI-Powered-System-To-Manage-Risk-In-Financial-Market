import pandas as pd
import time
import requests
import praw
from requests_oauthlib import OAuth1
import base64
import os
import yfinance as yf
import numpy as np
import joblib
from datetime import datetime, timedelta
import json
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import Json
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import asyncio
import mplfinance as mpf
import matplotlib.pyplot as plt
import psycopg2
import schedule
from pandas.plotting import table
from PIL import Image
import tempfile



#________________________________________ data extraction ________________________________________________________________________________________________________________________________________

start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
data = yf.download('AAPL', start=start_date, end=end_date, interval='60m')
data = data.drop(columns=['Adj Close', 'Volume'])
data.columns = data.columns.str.lower()

#________________________________________ data processing ________________________________________________________________________________________________________________________________________

def process_data(df):
    # Lag features and moving averages
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

    # MACD
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

    # Drop unnecessary columns
    columns_to_drop = ['open', 'high', 'low', 'date']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Drop the first 20 rows to account for the window period
    df = df.iloc[20:].reset_index(drop=True)
    

    return df

data_processed = process_data(data.copy())

#print(data)
#print(data_processed)
#results_file_path = "/content/data_processed.csv"
#data_processed.to_csv(results_file_path, index=False)
#print(f"Results saved to {results_file_path}.")


#________________________________________   model makes prediction ________________________________________________________________________________________________________________________________________


#GET THE PATH FRPM THE BEST MODULE FOLDER
model_path = "/usr/src/app/best_model_AAPL_60m_Lag10.joblib"
model = joblib.load(model_path, mmap_mode=None)

# Function to make predictions
def predict_currency_movement(data_processed):
    last_row = data_processed.iloc[-1:].to_numpy()
    print(last_row)
    prediction = model.predict(last_row)
    return prediction

#_________________________________________________ graph creation _________________________________________________________________________________________





def calculate_bollinger_bands(df, window_size=20, num_of_std=2):
    rolling_mean = df['close'].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df['close'].rolling(window=window_size, min_periods=1).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def fibonacci(df):
    phi = (1 + 5**0.5) / 2
    retracements = [0, 1/phi**2, 0.5, 1/phi, 1]
    return [(df['close'] - df['low'].min()) / (df['high'].max() - df['low'].min()) * level for level in retracements]

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal




# Visualization function
def visualize_stock(data, asset, time_frame, indicator):
    df = data.copy()
    apds = []
    if indicator == 'BB':
        upper_band, lower_band = calculate_bollinger_bands(df)
        apds.extend([mpf.make_addplot(upper_band, color='red', width=1.5), mpf.make_addplot(lower_band, color='green', width=1.5)])
    elif indicator == 'Fib':
        fib_levels = fibonacci(df)
        fib_colors = ['red', 'blue', 'green', 'orange', 'purple']
        apds.extend([mpf.make_addplot(level, color=color, width=0.75) for level, color in zip(fib_levels, fib_colors)])
    elif indicator == 'MACD':
        macd, signal = calculate_macd(df)
        apds.extend([mpf.make_addplot(macd, color='blue'), mpf.make_addplot(signal, color='orange')])

    s = mpf.make_mpf_style(base_mpf_style='nightclouds', rc={'axes.labelcolor': 'white'})
    image_path = tempfile.mktemp(suffix='.png')
    mpf.plot(df, type='candle', style=s, addplot=apds, title=f"{asset} {time_frame} - {indicator}", savefig=image_path)
    return image_path

# Generating and collecting temporary graph file paths
media_files = []  
indicators = ['BB', 'Fib', 'MACD']
for indicator in indicators:
    file_path = visualize_stock(data, 'GBPUSD', '90d', indicator)
    media_files.append(file_path)



##____________________________________________  posting functions _________________________________________________________________________________________________

#_________________________________________________ telegram psoting algorithem _________________________________________________________________________________________


def send_message_with_images(bot_token, chat_id, post_content, post_title ):
    formatted_content = f"<b>{post_title}</b>\n\n{post_content} check the graphs to create a more informed decision.\n\n not a fincacial advice"
    text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    text_params = {
        'chat_id': chat_id,
        'text': formatted_content,
        'parse_mode': 'HTML'  }
    text_response = requests.post(text_url, data=text_params)
    if text_response.status_code != 200:
        print(f"Error sending text: {text_response.content}")
    for image_path in media_files:
        with open(image_path, 'rb') as image:
            files = {'photo': image}
            response = requests.post(f"https://api.telegram.org/bot{bot_token}/sendPhoto", data={'chat_id': chat_id}, files=files)
            if response.status_code != 200:
                print(f"Error uploading image: {response.content}")

#_________________________________________________ twitter posting algorithem _________________________________________________________________________________________

api_key = 'lbwE3sP4GYeZlw8DRALIch6mA'
api_secret_key = '1IbBKePcLvEKGYo9MeBEW5UQhLCWhyBTU9jGkjUgcyZDi7RvPo'
access_token = '1779123882558943232-zLbcFIH1ZLdy7J10xPHntj5Wip2kcR'
access_token_secret = 'vSk212Cf1hesB0UU3g2bwnefPEDeGrrulR6EgoeFDrtQg'

def upload_media_oauth1(api_key, api_secret_key, access_token, access_token_secret, media_file):
    
    upload_url = "https://upload.twitter.com/1.1/media/upload.json"
    auth = OAuth1(api_key, api_secret_key, access_token, access_token_secret)
    with open(media_file, 'rb') as file:
        media_data = base64.b64encode(file.read()).decode('utf-8')
    response = requests.post(upload_url, auth=auth, data={'media_data': media_data})
    if response.status_code == 200:
        media_id = response.json()['media_id_string']
        return media_id
    else:
        print(f"Failed to upload media: {response.status_code}, {response.text}")
        return None

def post_tweet_with_media_oauth1(api_key, api_secret_key, access_token, access_token_secret, post_content_twitter, post_title, media_files):
    media_ids = []
    

    for media_file in media_files:
        media_id = upload_media_oauth1(api_key, api_secret_key, access_token, access_token_secret, media_file)
        if media_id:
            media_ids.append(media_id)
    if media_ids:
        tweet_url = "https://api.twitter.com/2/tweets"
        auth = OAuth1(api_key, api_secret_key, access_token, access_token_secret)
        formatted_content = f"{post_title}\n\n{post_content_twitter} check the graphs to create a more informed decision.\n\n not a fincacial advice" 
        payload = {"text": formatted_content, "media": {"media_ids": media_ids}}
        response = requests.post(tweet_url, auth=auth, json=payload)
        if response.status_code == 201:
            print("Tweet with media posted successfully!")
            return response.json()
        else:
            print(f"Failed to post tweet: {response.status_code}, {response.text}")
            return None
    else:
        print("Failed to upload media, tweet not posted.")
        return None

#_________________________________________________ linkedin posting algorithm _________________________________________________________________________________________


access_token_linkedin ='AQU8m-o7P50ClODmZEnQXogilN14uIZULI8zkT503GVWA5FlQs78fyEf6z7HApr6fYw2eGariVdFx5Q7g2Hn44uBmwyMF3Eg7gT4OhKlTT57Pjd-ZGBh4STRgBzokymOmEu95FN0Ibzab_hN694azFe2eWWRo9nNkXcMV3kXfXiISHKmM7uhFlwOohmnDLkEi95P1ZkkigwGzoN8KoJnnRjktv1JBYYHq_moy5xjcurn1_4zfWVKwpBOLmZO7aMLpZ5zFoFlyJbYodyeG0UcMXYLII-pOA1LprsopZTYpvCvJvmagXY1j1cSa-Zd3XFLlC5KbM36PYNjS0UxNWyNTXO-LTEIug'

def upload_image_to_linkedin(access_token_linkedin, image_path):
    register_upload_url = 'https://api.linkedin.com/v2/assets?action=registerUpload'
    headers = {
        'Authorization': f'Bearer {access_token_linkedin}',
        'Content-Type': 'application/json',
        'X-Restli-Protocol-Version': '2.0.0'
    }
    register_upload_data = {
        "registerUploadRequest": {
            "recipes": [
                "urn:li:digitalmediaRecipe:feedshare-image"
            ],
            "owner": "urn:li:person:UBoeSKj1P7",
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent"
                }
            ]
        }
    }
    response = requests.post(register_upload_url, headers=headers, json=register_upload_data)
    if response.status_code != 200:
        print(f"Failed to register image for upload. Status code: {response.status_code}, Response: {response.content}")
        return None
    upload_url = response.json()['value']['uploadMechanism']['com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest']['uploadUrl']
    asset = response.json()['value']['asset']
    with open(image_path, 'rb') as image_file:
        upload_response = requests.put(upload_url, headers={'Authorization': f'Bearer {access_token_linkedin}'}, data=image_file)
    if upload_response.status_code in [200, 201]:
        return asset 
    else:
        print(f"Failed to upload image. Status code: {upload_response.status_code}, Response: {upload_response.content}")
        return None

def create_post_with_image_linkedin(access_token_linkedin, post_title, post_content, assets):
    url = 'https://api.linkedin.com/v2/ugcPosts'
    headers = {
        'Authorization': f'Bearer {access_token_linkedin}',
        'X-Restli-Protocol-Version': '2.0.0',
        'Content-Type': 'application/json'
    }
    formatted_content = f"{post_title}\n\n{post_content} check the graphs to create a more informed decision.\n\n not a fincacial advice" 
    post_data = {
        "author": "urn:li:person:UBoeSKj1P7",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": formatted_content
                },
                "shareMediaCategory": "IMAGE",
                "media": assets
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }
    response = requests.post(url, headers=headers, json=post_data)
    if response.status_code == 201:
        print("")
    else:
        print(f"Failed to create post with image on LinkedIn. Status code: {response.status_code}, Response: {response.content}")


#_________________________________________________posting trigger and schdular function _________________________________________________________________________________________

 # Utility to clean up temporary files after posting
def cleanup_temp_files(file_list):
    for file_path in file_list:
        if os.path.exists(file_path):
            os.remove(file_path)
    
def post_to_social_media(reddit_posting=True, twitter=True, telegram=True, linkedin=True):
    # Reddit setup
    reddit = praw.Reddit(
        client_id='9_xhfty43vK-T-yYvpGtHA',
        client_secret='tIjyu5sPQj4n-pkCeAVRCzXhlDRYqA',
        password='dawdiCHTIAR2002',
        user_agent="python:AI_risk_manager:v1.0.0 (by /u/Strict_Analyst_8379)",
        username='Strict_Analyst_8379',
    )
    subreddit_name = "AI_RISK_PREDICTOR"

    # Telegram setup
    telegram_bot_token = '6730990236:AAF7wF-O0YilzbDL_csow30K2_s82EIQ-FA'
    telegram_chat_id = '-1002088223445'
    
    # GET THE PATH FOR THE POST MESSAGE FROM POSTING DATA FOLDER
    file_path = '/usr/src/app/posts_data.csv'
    df = pd.read_csv(file_path)
    
    prediction = predict_currency_movement(data_processed)
    print(f"Model Prediction: {prediction}")

    for index, row in df.iterrows():
        bullish_title = row['bullish_titles']
        bearish_title = row['bearish_titles']
        post_bearish = row['Description_bearish']
        post_bullish = row['Description_bullish']
        post_content_bearish = f"{post_bearish}"
        post_content_bullish = f"{post_bullish}"

        if reddit_posting:
            subreddit = reddit.subreddit(subreddit_name)
            if prediction == 1:
                subreddit.submit(bullish_title, selftext=post_content_bullish)
            else:  
                subreddit.submit(bearish_title, selftext=post_content_bearish)
            print('Posted to Reddit.')

        if twitter:
            try:
                if prediction == 1:
                    post_tweet_with_media_oauth1(api_key, api_secret_key, access_token, access_token_secret, post_content_bullish, bullish_title, media_files)
                else:
                    post_tweet_with_media_oauth1(api_key, api_secret_key, access_token, access_token_secret, post_content_bearish, bearish_title, media_files)
                print('Posted to Twitter.')
            except Exception as e:
                print(f"Failed to post to Twitter: {str(e)}")

        if telegram:
            if prediction == 1:
                send_message_with_images(telegram_bot_token, telegram_chat_id, post_content_bullish,bullish_title)
            else:
                send_message_with_images(telegram_bot_token, telegram_chat_id, post_content_bearish,bearish_title)
            print('Posted to Telegram.')

        if linkedin:
            assets = []
            for image_path in media_files:
                asset = upload_image_to_linkedin(access_token_linkedin, image_path)
                if asset:
                    assets.append({"media": asset, "status": "READY"})
            if assets:
                create_post_with_image_linkedin(access_token_linkedin, bullish_title if prediction == 1 else bearish_title, post_content_bullish if prediction == 1 else post_content_bearish, assets)
                print('Posted to LinkedIn.')
            else:
                print("Failed to upload images, cannot create post on LinkedIn.")
                
            
        print("finished posting for today , posting in the next 24 hours...")
                
            
        time.sleep(86400)
post_to_social_media(reddit_posting=False, twitter=False, telegram=True, linkedin=False)
cleanup_temp_files(media_files[1:])
    


