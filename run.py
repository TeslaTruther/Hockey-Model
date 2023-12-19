import streamlit as st
import pandas as pd
import scipy.stats as stats
from datetime import datetime

st.set_page_config(page_title="Free Hockey Locks", page_icon=":lock:", layout="wide")  

# Header section
st.markdown("### Hockey Locks :lock:")
st.markdown("Last Data Update: 12/19/23")   
st.write("Shoutout Cam for always riding")
st.write("How the model works: Compare these odds to your sportsbook's odds. If my projected odds are lower than the sportsbook's odds place the bet.")  
st.write("Things to consider: The projected odds do not include rake so will often be higher then your sportsbook showing no value. The projected odds do not consider back to back or injuries which could show false value.")

# Use a relative path to the Excel file
excel_file = '2024 Schedule.xlsx'

# Load the Excel file containing game data and wins data
df = pd.read_excel(excel_file, sheet_name="Schedule")
wins_data = pd.read_excel(excel_file, sheet_name="Test Data")

# Get today's date dynamically
today = datetime.now().date()

# Convert 'today' to a Pandas Timestamp object
today_timestamp = pd.Timestamp(today)

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


# Button to get today's odds
if st.button("Get Today's Odds", key="get_today_odds"):
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
    st.write("## Odds for Today's Games:")
    for i, game in enumerate(game_odds, start=1):
        st.write(f"### Game {i}:")
        st.write(f"**Home Team:** {game['Home Team']} | **Projected Odds:** {game['Decimal Win Odds']:.3f}")
        st.write(f"**Visitor Team:** {game['Visitor Team']} | **Projected Odds:** {game['Decimal Lose Odds']:.3f}")
        st.write("\n")

# Button to get tomorrow's odds
if st.button("Get Tomorrow's Odds", key="get_tomorrow_odds"):
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
    st.write("## Odds for Tomorrow's Games:")
    for i, game in enumerate(game_odds, start=1):
        st.write(f"### Game {i}:")
        st.write(f"**Home Team:** {game['Home Team']} | **Projected Odds:** {game['Decimal Win Odds']:.3f}")
        st.write(f"**Visitor Team:** {game['Visitor Team']} | **Projected Odds:** {game['Decimal Lose Odds']:.3f}")
        st.write("\n")
