import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from PIL import Image
import os
import scipy.stats as stats

st.set_page_config(page_title="Hockey Locks", page_icon="🔒", layout="wide")

st.markdown('<link rel="stylesheet" type="text/css" href="style.css">', unsafe_allow_html=True)

class LazyImage:
    def __init__(self, path):
        self.path = path
        self.image = None

    def load(self):
        if self.image is None:
            self.image = Image.open(self.path)

    def resize(self, size):
        self.load()
        return self.image.resize(size)

panda_folder = 'images'

# List all files in the specified folder
panda_files = os.listdir(panda_folder)

# Create LazyImage instances for each panda picture
lazy_pandas = [LazyImage(os.path.join(panda_folder, file)) for file in panda_files]

# Resize the images only when needed
resized_pandas = [lazy_panda.resize((400, 400)) for lazy_panda in lazy_pandas]

# Function to adjust date based on time zone offset
def adjust_date_by_offset(date_column, time_zone_offset):
    return date_column + pd.to_timedelta(time_zone_offset, unit='hours')

# Set the time zone to Pacific Time (UTC-8)
time_zone = pytz.timezone('America/Los_Angeles')

# Use a relative path to the Excel file
excel_file = '2024 Schedule.xlsx'

# Load data from "Game Data" sheet
game_data = pd.read_excel(excel_file, sheet_name="test")

# Convert 'Date' column to datetime format and adjust for Pacific Time
game_data['Date'] = pd.to_datetime(game_data['Date']).dt.tz_localize(time_zone)

# Get today's date dynamically in Pacific Time
today = datetime.now(time_zone).replace(hour=0, minute=0, second=0, microsecond=0)

# Filter the DataFrame to get today's games
today_games = game_data[(game_data['Date'] >= today) & (game_data['Date'] < today + pd.DateOffset(1))]

# Get tomorrow's date dynamically in Pacific Time
tomorrow = today + timedelta(days=1)
tomorrow_start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
tomorrow_end = tomorrow_start + pd.DateOffset(1)
tomorrow_games = game_data[(game_data['Date'] >= tomorrow_start) & (game_data['Date'] < tomorrow_end)]

# Sidebar with a smaller width
selection = st.sidebar.radio('Hockey Locks 🔒', ['Home', 'NHL Model', 'Performance Tracking'])

if selection == 'Home':
    # Main content
    st.title("Welcome to Hockey Locks :lock:")
    st.write("This model calculates expected hockey odds by regressing a series of analytics over the last 2 NHL seasons.")
    st.write("Use this tool to find inefficient odds and make positive EV bets.")   
     
    st.image(resized_pandas[2])      
elif selection == 'NHL Model':
    st.header("How the Model Works")
    st.write("Compare these odds to your sportsbook's odds. If my projected odds are lower than the sportsbook's odds, place the bet.")
    st.subheader("Model Considerations")
    st.write("The model does not include rake.")
    st.write("The model does not incorporate back-to-back games.")
    st.subheader("Run The Model:")
    

# Button to get today's odds
# Button to get today's odds
    if st.button("Generate Today's Odds", key="get_today_odds"):
        # Calculate and display the over/under odds, implied probabilities, and projected scores
        today_games['Projected_Score'] = today_games['hometotal'] + today_games['vistotal']

        # Calculate the projected Money Line odds
        today_games['Projected_Line'] = 0.55 * today_games['ml1'] + 0.45 * today_games['ml2']

        # Round the constant to the nearest 0.5 using round_half_even
        today_games['Constant'] = np.round(today_games['Projected_Score'] / 0.5) * 0.5

        # Set the standard deviation
        std_deviation_overunder = 2.0
        std_deviation_ml = 2.5

        # Calculate implied prob for ML
        today_games['ML_Home_Prob'] = today_games.apply(
            lambda row: stats.norm.cdf((row.Projected_Line) / std_deviation_ml),
            axis=1
        )

        today_games['ML_Away_Prob'] = today_games.apply(
            lambda row: stats.norm.cdf(- (row.Projected_Line) / std_deviation_ml),
            axis=1
        )

        # Convert implied probabilities to decimal odds for ML
        today_games['ML_Home_Decimal_Odds'] = 1 / today_games['ML_Home_Prob']
        today_games['ML_Away_Decimal_Odds'] = 1 / today_games['ML_Away_Prob']


        # Calculate the odds for over/under using the normal distribution
        today_games['Over_Under_Odds'] = today_games.apply(
            lambda row: {
                'Over': 1 - stats.norm.cdf((row.Constant - row.Projected_Score) / std_deviation_overunder),
                'Under': stats.norm.cdf((row.Constant - row.Projected_Score) / std_deviation_overunder)
            },
            axis=1
        )

        # Calculate the implied probability percentages for Over/Under
        today_games['Totals_Probability'] = today_games['Over_Under_Odds'].apply(
            lambda odds: {'Over': 1 / odds['Over'], 'Under': 1 / odds['Under']}
        )

        # Calculate decimal odds for Over/Under
        today_games['Totals_Decimal_Odds'] = today_games['Totals_Probability'].apply(
            lambda odds: {'Over': odds['Over'] - 1, 'Under': odds['Under'] - 1}
        )

             # Display the odds for today's games in a Streamlit table
        # Display the odds for today's games in a Streamlit table
        st.write("### Today's Games and Projected Odds:")
        for i, game in enumerate(today_games.itertuples(), start=1):                   
            st.subheader(f"{game.Visitor} *@* {game.Home}")
            st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
            st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")
            
            st.write(f"Projected Over Under Line: {game.Constant:.1f}")            
            st.write(f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")

            
    if st.button("Generate Tomorrow's Odds", key="get_tomorrows_odds"):
         
         # Calculate and display the over/under odds, implied probabilities, and projected scores
        tomorrow_games['Projected_Score'] = tomorrow_games['hometotal'] + tomorrow_games['vistotal']

        # Calculate the projected Money Line odds
        tomorrow_games['Projected_Line'] = 0.55 * tomorrow_games['ml1'] + 0.45 * tomorrow_games['ml2']

        # Round the constant to the nearest 0.5 using round_half_even
        tomorrow_games['Constant'] = np.round(tomorrow_games['Projected_Score'] / 0.5) * 0.5

        # Set the standard deviation
        std_deviation_overunder = 2.0
        std_deviation_ml = 2.5

        # Calculate implied prob for ML
        tomorrow_games['ML_Home_Prob'] = tomorrow_games.apply(
            lambda row: stats.norm.cdf((row.Projected_Line) / std_deviation_ml),
            axis=1
        )

        tomorrow_games['ML_Away_Prob'] = tomorrow_games.apply(
            lambda row: stats.norm.cdf(- (row.Projected_Line) / std_deviation_ml),
            axis=1
        )

        # Convert implied probabilities to decimal odds for ML
        tomorrow_games['ML_Home_Decimal_Odds'] = 1 / tomorrow_games['ML_Home_Prob']
        tomorrow_games['ML_Away_Decimal_Odds'] = 1 / tomorrow_games['ML_Away_Prob']




        # Calculate the odds for over/under using the normal distribution
        tomorrow_games['Over_Under_Odds'] = tomorrow_games.apply(
            lambda row: {
                'Over': 1 - stats.norm.cdf((row.Constant - row.Projected_Score) / std_deviation_overunder),
                'Under': stats.norm.cdf((row.Constant - row.Projected_Score) / std_deviation_overunder)
            },
            axis=1
        )

        # Calculate the implied probability percentages for Over/Under
        tomorrow_games['Totals_Probability'] = tomorrow_games['Over_Under_Odds'].apply(
            lambda odds: {'Over': 1 / odds['Over'], 'Under': 1 / odds['Under']}
        )

        # Calculate decimal odds for Over/Under
        tomorrow_games['Totals_Decimal_Odds'] = tomorrow_games['Totals_Probability'].apply(
            lambda odds: {'Over': odds['Over'] - 1, 'Under': odds['Under'] - 1}
        )

             # Display the odds for today's games in a Streamlit table
        # Display the odds for today's games in a Streamlit table
        st.write("### Tomorrow's Games and Projected Odds:")
           
        for i, game in enumerate(tomorrow_games.itertuples(), start=1):                   
            st.subheader(f"{game.Visitor} *@* {game.Home}")
            st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
            st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")
            
            st.write(f"Projected Over Under Line: {game.Constant:.1f}")            
            st.write(f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")
            
         
            
            


# Display the odds for today's games in a Streamlit table

elif selection == 'Performance Tracking':
    st.subheader('ACCUMULATING GAINS - Coming soon')
    st.image(resized_pandas[0])