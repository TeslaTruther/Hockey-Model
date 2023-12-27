import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Hockey Locks", page_icon="🔒", layout="wide")

# Custom CSS for styling the sidebar
custom_sidebar_css = """
    <style>
        .sidebar .sidebar-content {
            width: 200px;  /* Adjust the width as needed */
            background-color: #333;  /* Sidebar background color */
        }

        .sidebar .sidebar-content .stRadio {
            color: white;  /* Radio button text color */
        }
    </style>
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Hockey Locks", page_icon="🔒", layout="wide")

# Custom CSS for styling the sidebar
custom_sidebar_css = """
    <style>
        .sidebar .sidebar-content {
            width: 200px;  /* Adjust the width as needed */
            background-color: #333;  /* Sidebar background color */
        }

        .sidebar .sidebar-content .stRadio {
            color: white;  /* Radio button text color */
        }
    </style>
"""

# Apply custom CSS to the sidebar
st.markdown(custom_sidebar_css, unsafe_allow_html=True)

# Function to adjust date based on time zone offset
def adjust_date_by_offset(date_column, time_zone_offset):
    return date_column + pd.to_timedelta(time_zone_offset, unit='h')

# Set the time zone offset to 8 hours back (UTC-8)
selected_time_zone_offset = -8

# Use a relative path to the Excel file
excel_file = '2024 Schedule.xlsx'

# Load the Excel file containing game data and wins data
df = pd.read_excel(excel_file, sheet_name="Schedule")
wins_data = pd.read_excel(excel_file, sheet_name="Test Data")

# Convert 'Date' column to datetime format and adjust for the time zone offset
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = adjust_date_by_offset(df['Date'], selected_time_zone_offset)

# Get today's date dynamically
today = datetime.now().date()

# Convert 'today' to a Pandas Timestamp object and adjust for the selected time zone offset
today_timestamp = pd.Timestamp(today) + timedelta(hours=selected_time_zone_offset)

# Filter the DataFrame to get today's games
today_games = df[(df['Date'] >= today_timestamp) & (df['Date'] < today_timestamp + pd.DateOffset(1))]

tomorrow_timestamp = today_timestamp + pd.DateOffset(1)
tomorrow_games = df[(df['Date'] >= tomorrow_timestamp) & (df['Date'] < tomorrow_timestamp + pd.DateOffset(1))]



def determine_winner(home_team, visitor_team, wins_data, home_goal_advantage=0.18):
    home_srs = pd.to_numeric(wins_data[wins_data['Team'] == home_team]['Average SRS'], errors='coerce')
    home_test4 = pd.to_numeric(wins_data[wins_data['Team'] == home_team]['Test 4'], errors='coerce')
    home_test6 = pd.to_numeric(wins_data[wins_data['Team'] == home_team]['Test 6'], errors='coerce')

    visitor_srs = pd.to_numeric(wins_data[wins_data['Team'] == visitor_team]['Average SRS'], errors='coerce')
    visitor_test4 = pd.to_numeric(wins_data[wins_data['Team'] == visitor_team]['Test 4'], errors='coerce')
    visitor_test6 = pd.to_numeric(wins_data[wins_data['Team'] == visitor_team]['Test 6'], errors='coerce')

    # Check if both stats are valid
    if (home_srs.notna().all() and home_test4.notna().all() and home_test6.notna().all() and
            visitor_srs.notna().all() and visitor_test4.notna().all() and visitor_test6.notna().all()):
        home_srs = home_srs.item()
        home_test4 = home_test4.item()
        home_test6 = home_test6.item()

        visitor_srs = visitor_srs.item()
        visitor_test4 = visitor_test4.item()
        visitor_test6 = visitor_test6.item()

        # Adjust the goal difference to give the home team a 0.15 goal advantage
        expected_goal_diff = 0 * (home_srs - visitor_srs) + 0.55 * (home_test4 - visitor_test4) + 0.45 * (
                home_test6 - visitor_test6) + home_goal_advantage

        if expected_goal_diff > 0:
            winner = home_team
        elif expected_goal_diff < 0:
            winner = visitor_team
        else:
            winner = 'Tie'
    else:
        winner = 'Tie'

    return winner, expected_goal_diff



import scipy.stats as stats

def implied_probabilities(margin_of_victory):
    # Standard deviation for margin of victory (you can adjust this)
    std_dev = 2.48

    # Calculate implied probabilities for win, lose, and tie
    win_prob = 1 - stats.norm.cdf(0, loc=margin_of_victory, scale=std_dev)
    lose_prob = stats.norm.cdf(0, loc=margin_of_victory, scale=std_dev)
    tie_prob = 1 - win_prob - lose_prob

    return win_prob, lose_prob, tie_prob


def implied_probability_to_decimal_odds(probability):
    if probability > 0:
        decimal_odds = 1 / probability
    else:
        decimal_odds = 0
    return decimal_odds


game_odds = []

# Apply custom CSS to the sidebar
st.markdown(custom_sidebar_css, unsafe_allow_html=True)

# Sidebar with a smaller width
selection = st.sidebar.radio('Hockey Locks 🔒', ['Home', 'Hockey Model', 'Performance Tracking'])

if selection == 'Home':
    # Main content
    st.title("Welcome to Hockey Locks :lock:")
    st.write("This model calculates expected hockey odds by regressing a series of analytics over the last 2 NHL seasons.")
    st.write("Use this tool to catch your bookie sleeping and make positive EV hockey bets.")   
      
elif selection == 'Hockey Model':

    st.header("How the Model Works")
    st.write("Compare these odds to your sportsbook's odds. If my projected odds are lower than the sportsbook's odds, place the bet.")
    st.subheader("Model Considerations")
    st.write("The model does not include rake by design.")
    st.write("The model does not incorporate back-to-back games or injuries. These statistically significant varibles need to be adjusted manually.")
    st.subheader("Run The Model:")
    # Button to get today's odds
    if st.button("Generate Today's Odds", key="get_today_odds"):
        for index, row in today_games.iterrows():
            home_team = row['Home']
            visitor_team = row['Visitor']

            winner, delta = determine_winner(home_team, visitor_team, wins_data, home_goal_advantage=0.18)

            # Calculate the implied probabilities and odds
            margin_of_victory = delta
            win_prob, lose_prob, tie_prob = implied_probabilities(margin_of_victory)

            # Convert adjusted probabilities to decimal odds
            decimal_win_odds = implied_probability_to_decimal_odds(win_prob)
            decimal_lose_odds = implied_probability_to_decimal_odds(lose_prob)

            game_odds.append({
                'Home Team': home_team,
                'Visitor Team': visitor_team,
                'Winner': winner,
                'Delta': delta,
                'Implied Win Probability': win_prob,
                'Implied Lose Probability': lose_prob,
                'Decimal Win Odds': decimal_win_odds,
                'Decimal Lose Odds': decimal_lose_odds
            })

        # Display the odds for today's games in a Streamlit table
        
        # Display the odds for today's games in a Streamlit table

        for i, game in enumerate(game_odds, start=1):
            st.write(f"### Game {i}:")
            st.write(f"**Home Team:** {game['Home Team']} | **Projected Odds:** {game['Decimal Win Odds']:.3f}")
            st.write(f"**Visitor Team:** {game['Visitor Team']} | **Projected Odds:** {game['Decimal Lose Odds']:.3f}")
            st.write("\n")

    # Button to get tomorrow's odds
    if st.button("Generate Tomorrow's Odds", key="get_tomorrow_odds"):
        game_odds = []  # Reset the list for tomorrow's odds
        for index, row in tomorrow_games.iterrows():
            home_team = row['Home']
            visitor_team = row['Visitor']

            winner, delta = determine_winner(home_team, visitor_team, wins_data, home_goal_advantage=0.18)

            # Calculate the implied probabilities and odds
            margin_of_victory = delta
            win_prob, lose_prob, tie_prob = implied_probabilities(margin_of_victory)

            # Convert adjusted probabilities to decimal odds
            decimal_win_odds = implied_probability_to_decimal_odds(win_prob)
            decimal_lose_odds = implied_probability_to_decimal_odds(lose_prob)

            game_odds.append({
                'Home Team': home_team,
                'Visitor Team': visitor_team,
                'Winner': winner,
                'Delta': delta,
                'Implied Win Probability': win_prob,
                'Implied Lose Probability': lose_prob,
                'Decimal Win Odds': decimal_win_odds,
                'Decimal Lose Odds': decimal_lose_odds
            })

        # Display the odds for tomorrow's games in a Streamlit table
        
        for i, game in enumerate(game_odds, start=1):
            st.write(f"### Game {i}:")
            st.write(f"**Home Team:** {game['Home Team']} | **Projected Odds:** {game['Decimal Win Odds']:.3f}")
            st.write(f"**Visitor Team:** {game['Visitor Team']} | **Projected Odds:** {game['Decimal Lose Odds']:.3f}")
            st.write("\n")

        

elif selection == 'Performance Tracking':
    st.write('ACCUMULATING GAINS - Coming soon')



