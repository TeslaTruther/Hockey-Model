import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz  
from PIL import Image 
import scipy.stats as stats

# Function to adjust date based on time zone offset
def adjust_date_by_offset(date_column, time_zone_offset):
    return date_column + pd.to_timedelta(time_zone_offset, unit='hours')


# Set the time zone to Pacific Time (UTC-8)
time_zone = pytz.timezone('America/Los_Angeles')

# Use a relative path to the Excel file
excel_file = '2024 Schedule.xlsx'

# Load the Excel file containing game data and wins data
df = pd.read_excel(excel_file, sheet_name="Schedule")
wins_data = pd.read_excel(excel_file, sheet_name="MoneyLine")

# Convert 'Date' column to datetime format and adjust for Pacific Time
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(time_zone)

# Get today's date dynamically in Pacific Time
today = datetime.now(time_zone).replace(hour=0, minute=0, second=0, microsecond=0)

# Filter the DataFrame to get today's games
today_games = df[(df['Date'] >= today) & (df['Date'] < today + pd.DateOffset(1))]

# Get tomorrow's date dynamically in Pacific Time
tomorrow = today + timedelta(days=1)
tomorrow_start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
tomorrow_end = tomorrow_start + pd.DateOffset(1)
tomorrow_games = df[(df['Date'] >= tomorrow_start) & (df['Date'] < tomorrow_end)]




def determine_score(home_team, visitor_team, wins_data):
    home_total = pd.to_numeric(wins_data[wins_data['Team'] == home_team]['hometotal'], errors='coerce')
    visitor_total = pd.to_numeric(wins_data[wins_data['Team'] == visitor_team]['vistotal'], errors='coerce')

    # Check if both stats are valid
    if home_total.notna().all() and visitor_total.notna().all():
        home_total = home_total.item()
        visitor_total = visitor_total.item()

        # Adjust the goal difference to give the home team a 0.15 goal advantage
        expected_total = home_total + visitor_total

        return expected_total
    else:
        return 'Invalid Data'
# Function to calculate implied probabilities and odds for over/under on 6.5 goals
def calculate_over_under_odds(expected_total, goal_threshold=6.5):
    margin_of_victory = expected_total - goal_threshold

    # Calculate implied probabilities for over, under, and push
    over_prob = 1 - stats.norm.cdf(0, loc=margin_of_victory, scale=1.7)
    under_prob = stats.norm.cdf(0, loc=margin_of_victory, scale=1.7)
    push_prob = 1 - over_prob - under_prob

    # Convert probabilities to decimal odds
    over_odds = implied_probability_to_decimal_odds(over_prob)
    under_odds = implied_probability_to_decimal_odds(under_prob)

    return over_prob, under_prob, push_prob, over_odds, under_odds

def implied_probability_to_decimal_odds(probability):
    if probability > 0:
        decimal_odds = 1 / probability
    else:
        decimal_odds = 0
    return decimal_odds

# Calculate and print over/under odds for tomorrow's games
for index, row in tomorrow_games.iterrows():
    home_team = row['Home']
    visitor_team = row['Visitor']

    expected_total = determine_score(home_team, visitor_team, wins_data)

    over_prob, under_prob, push_prob, over_odds, under_odds = calculate_over_under_odds(expected_total)

    print(f"{home_team} vs {visitor_team}:")
    print(f"Over {over_odds:.2f}")
    print(f"Under {under_odds:.2f}")
    print("\n")