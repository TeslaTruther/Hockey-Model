import pandas as pd
import scipy.stats as stats
import numpy as np
from datetime import datetime
import streamlit as st

# Load the Excel file containing game data and wins data
excel_file = '2024 Schedule.xlsx'

# Load data from "Game Data" sheet
game_data = pd.read_excel(excel_file, sheet_name="test")

# Get tomorrow's date dynamically
tomorrow = datetime.now().date() + pd.DateOffset(1)

# Convert 'tomorrow' to a Pandas Timestamp object
tomorrow_timestamp = pd.Timestamp(tomorrow)

# Filter the DataFrame to get today's games
tomorrow_games = game_data[(game_data['Date'] >= tomorrow_timestamp) & (game_data['Date'] < tomorrow_timestamp + pd.DateOffset(1))]

# Check if there are games tomorrow
if tomorrow_games.empty:
    st.write("No games tomorrow.")
    
else:
    # Calculate the 'Projected_Score' as the sum of 'hometotal' and 'vistotal'
    tomorrow_games['Projected_Score'] = tomorrow_games['hometotal'] + tomorrow_games['vistotal']

    # Round the constant to the nearest 0.5 using round_half_even
    tomorrow_games['Constant'] = np.round(tomorrow_games['Projected_Score'] / 0.5) * 0.5

    # Set the standard deviation
    std_deviation = 2

    # Calculate the odds for over/under using the normal distribution
    tomorrow_games['Over_Under_Odds'] = tomorrow_games.apply(
        lambda row: {
            'Over': 1 - stats.norm.cdf((row.Constant - row.Projected_Score) / std_deviation),
            'Under': stats.norm.cdf((row.Constant - row.Projected_Score) / std_deviation)
        },
        axis=1
    )

    # Calculate the implied probability percentages
    tomorrow_games['Implied_Probability'] = tomorrow_games['Over_Under_Odds'].apply(
        lambda odds: {'Over': 1 / odds['Over'], 'Under': 1 / odds['Under']}
    )

    # Calculate decimal odds
    tomorrow_games['Decimal_Odds'] = tomorrow_games['Implied_Probability'].apply(
        lambda odds: {'Over': odds['Over'] - 1, 'Under': odds['Under'] - 1}
    )

    # Display the odds for tomorrow's games in a Streamlit table
    st.write("### Tomorrow's Games and Projected Scores:")
    for i, game in enumerate(tomorrow_games.itertuples(), start=1):
        st.write(f" *Game {i}*:")
        st.write(f"{game.Visitor} *@* {game.Home}  \n**Projected Over Under Line:** {game.Constant:.1f}")
        st.write(f"**Odds:** Over: {game.Implied_Probability['Over']:.2f}, Under: {game.Implied_Probability['Under']:.2f}")
        st.write(f"**Projected Score:** {game.Projected_Score:.2f}")
        
