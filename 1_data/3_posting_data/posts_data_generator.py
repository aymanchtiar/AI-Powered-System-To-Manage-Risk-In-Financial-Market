import csv
import random

bearish_titles = ([
    "AAPL: Bearish Momentum Expected",
    "AI Predicts Downturn for AAPL",
    "Bearish Outlook: AAPL Forecast",
    "Prepare for Decline: AAPL Prediction",
    "Caution Advised: Bearish AAPL Signal",
    "Downward Pressure: AAPL Analysis",
    "Watch Out: AAPL Could Drop",
    "Bearish Signal Detected: AAPL Analysis",
    "Potential Downtrend: AAPL Prediction",
    "Forecast: AAPL Likely to Decrease",
    "Bearish Trends: AAPL Analysis",
    "Warning Signs: AAPL Bearish Forecast",
    "Bearish Momentum: AAPL Projection",
    "Potential Bearish Movement: AAPL Prediction",
    "AAPL Forecast: Negative Outlook",
    "AAPL Analysis: Signs of Bearishness",
    "Potential Downturn: AAPL Forecast",
    "Prepare for Lower AAPL: AI Prediction",
    "Watch for Bearish Trends: AAPL Forecast",
    "AAPL Analysis: Expecting Bearish Phase"
])

bullish_titles = ([
    "AAPL: Bullish Trend on the Horizon",
    "AI Forecasts Upward Movement for AAPL",
    "Positive Outlook: Bullish AAPL Prediction",
    "Optimism Ahead: AAPL Bullish Forecast",
    "Upward Momentum Expected: AAPL Prediction",
    "Bullish Sentiment: AAPL Analysis",
    "Expect Growth: AAPL Could Rise",
    "Bullish Signal Detected: AAPL Analysis",
    "Potential Uptrend: AAPL Prediction",
    "Forecast: AAPL Likely to Increase",
    "Bullish Trends: AAPL Analysis",
    "Optimistic Signals: AAPL Bullish Forecast",
    "Bullish Momentum: AAPL Projection",
    "Potential Bullish Movement: AAPL Prediction",
    "AAPL Forecast: Positive Outlook",
    "AAPL Analysis: Signs of Bullishness",
    "Potential Upswing: AAPL Forecast",
    "Prepare for Higher AAPL: AI Prediction",
    "Watch for Bullish Trends: AAPL Forecast",
    "AAPL Analysis: Expecting Bullish Phase"
])

Description_bearish = ([
    "AI module anticipates a bearish trajectory for AAPL over the next 10 hours.",
    "Expect AAPL to trend downwards, AI analysis indicates, for the next 10 hours.",
    "Bearish sentiment dominates AI's AAPL forecast for the upcoming 10 hours.",
    "AI predicts a decline for AAPL in the next 10 hours.",
    "AAPL likely to experience a bearish trend, AI projection suggests, over the next 10 hours.",
    "Pressure mounts as AAPL faces potential downtrend, AI suggests.",
    "Be cautious as AAPL shows signs of potential decline, according to AI analysis.",
    "AI identifies a bearish signal in AAPL trajectory for the next 10 hours.",
    "Prepare for possible downtrend in AAPL, AI forecasts.",
    "AI analysis indicates potential decrease in AAPL value over the next 10 hours.",
    "AI predicts bearish momentum for AAPL in the upcoming 10 hours.",
    "Be wary of downward movement in AAPL, AI warns.",
    "Watch out for bearish momentum in AAPL, AI suggests.",
    "AI detects potential for bearish movement in AAPL, prepare accordingly.",
    "AI forecasts negative outlook for AAPL over the next 10 hours.",
    "AI analysis indicates signs pointing towards a bearish phase for AAPL.",
    "Prepare for potential downturn in AAPL value, AI forecasts.",
    "AI suggests lower AAPL values in the near future, prepare for impact.",
    "AI advises vigilance against bearish trends in AAPL, monitor closely.",
    "AI analysis suggests AAPL may enter a bearish phase soon."
])

Description_bullish = ([
    "AI module foresees a bullish trajectory for AAPL over the next 10 hours.",
    "Expect AAPL to trend upwards, AI analysis indicates, for the next 10 hours.",
    "Bullish sentiment prevails in AI's AAPL forecast for the upcoming 10 hours.",
    "AI predicts an upward movement for AAPL in the next 10 hours.",
    "AAPL likely to experience a bullish trend, AI projection suggests, over the next 10 hours.",
    "Bullish momentum builds as AI forecasts positive trajectory for AAPL.",
    "Expect growth in AAPL as AI detects bullish signals in the next 10 hours.",
    "AI identifies a bullish signal in AAPL trajectory for the next 10 hours.",
    "Prepare for potential uptrend in AAPL, AI forecasts.",
    "AI analysis indicates potential increase in AAPL value over the next 10 hours.",
    "AI predicts bullish momentum for AAPL in the upcoming 10 hours.",
    "Anticipate upward movement in AAPL, AI suggests.",
    "Watch out for bullish momentum in AAPL, AI advises.",
    "AI detects potential for bullish movement in AAPL, prepare accordingly.",
    "AI forecasts positive outlook for AAPL over the next 10 hours.",
    "AI analysis indicates signs pointing towards a bullish phase for AAPL.",
    "Prepare for potential upswing in AAPL value, AI forecasts.",
    "AI suggests higher AAPL values in the near future, prepare for impact.",
    "AI advises vigilance against bullish trends in AAPL, monitor closely.",
    "AI analysis suggests AAPL may enter a bullish phase soon."
])

bearish_posts = [(description, random.choice(bearish_titles)) for description in Description_bearish]
bullish_posts = [(description, random.choice(bullish_titles)) for description in Description_bullish]

# Write to CSV file
with open('posts_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Description_bearish', 'bearish_titles', 'Description_bullish', 'bullish_titles'])
    for bearish_post, bullish_post in zip(bearish_posts, bullish_posts):
        writer.writerow([bearish_post[0], bearish_post[1], bullish_post[0], bullish_post[1]])

print("CSV file 'posts_data.csv' has been created successfully!")
