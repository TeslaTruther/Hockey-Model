from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from PIL import Image
import os
import scipy.stats as stats
import plotly.express as px
import base64
import requests
from io import BytesIO
from nba_api.stats.endpoints import leaguestandingsv3
st.set_page_config(page_title="Quantum Odds", page_icon="🔒", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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



# Sidebar with a smaller width
selection = st.sidebar.radio('Quantum Odds 	✅', ['🏠 Home','🏀 NBA Model','🏀 NBA Franchise Mode', '🏒 NHL Model', '🔑 Betting Strategy', '💲Performance Tracking'])

if selection == '🏠 Home':
    # Main content
    st.title("Quantum Odds 	✅")
    st.write("We generate odds so you can compete against sportsbooks.")
    st.write("Find inefficient markets and make positive expected value bets.")   
     
    st.image(resized_pandas[0])    



 
elif selection == '🏒 NHL Model':
           
       
        excel_file = 'nhl.xlsx'
        def fetch_fanduel_nhl_odds():
            # Your Odds API key
            API_KEY = "0b8b0f798933d0c1e0ba7a7228ec21fa" 

            # Endpoint for NHL odds
            url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"

            # Parameters for the API request
            params = {
                "apiKey": API_KEY,
                "regions": "us",  # Focus on US bookmakers
                "markets": "h2h,totals",  # Get moneyline and over/under odds
                "oddsFormat": "decimal"  # Request odds in decimal format
            }

            # Send the API request
            response = requests.get(url, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()

                # Check if any odds are found
                if data:
                    nhl_odds = []
                    for game in data:
                        home_team = game['home_team']
                        away_team = game['away_team']
                        commence_time = game['commence_time']
                        total_line = None
                        total_written = False  # Flag to track if total line has been written

                        # Find bookmakers and their odds
                        bookmakers = game['bookmakers']
                        for bookmaker in bookmakers:
                            # Focus on FanDuel bookmaker
                            if bookmaker['title'] == 'FanDuel':
                                markets = bookmaker['markets']
                                for market in markets:
                                    # Check for moneyline odds
                                    if market['key'] == 'h2h':
                                        home_win_odds = market['outcomes'][0]['price']
                                        away_win_odds = market['outcomes'][1]['price']
                                    # Check for over/under odds
                                    elif market['key'] == 'totals':
                                        if not total_written:  # Write total line only once
                                            total_line = f"{market['outcomes'][0]['point']}"
                                            over_odds = market['outcomes'][0]['price']
                                            under_odds = market['outcomes'][1]['price']
                                            total_written = True

                        nhl_odds.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': commence_time,
                            'home_win_odds': home_win_odds,
                            'away_win_odds': away_win_odds,
                            'over_odds': over_odds,
                            'under_odds': under_odds,
                            'total_line': total_line
                        })
                    return nhl_odds
                else:
                    pass
                    return []
            else:
                st.error("Request failed. Status code:", response.status_code)
                return []
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
        def find_last_matchup(team1, team2, data, today):
            yesterday = today - timedelta(days=1)
            last_matchup_date = None
            last_matchup_scores = None
            
            # Check if there are any previous matchups between the teams
            matchups = ((data['Visitor'] == team1) & (data['Home'] == team2) | (data['Visitor'] == team2) & (data['Home'] == team1))
            if matchups.any():
                # Iterate over the DataFrame in reverse chronological order
                for index, row in data.loc[matchups].sort_values(by='Date', ascending=False).iterrows():
                    # Exclude today's games
                    if row['Date'] != today:
                        last_matchup_date = row['Date']
                        if row['Visitor'] == team1:
                            last_matchup_scores = (row['G2'], row['G'])
                        else:
                            last_matchup_scores = (row['G'], row['G2'])
                        if last_matchup_date <= yesterday:  # Check if the last matchup date is on or before yesterday
                            break  # Break out of the loop after finding the last matchup

            return last_matchup_date, last_matchup_scores



        st.title('NHL Model 🏒 ')
        st.header("How the Model Works")
        st.write("The model generates odds from its projected probability of outcomes. Think of these odds as the minimum return you would require to make a positive EV bet.")
        # Define a list of available methods for calculating odds
        ##calculation_methods = ['Decimal', 'American']

        # Add a selectbox to choose the calculation method
       ## selected_method = st.selectbox('Select Odds:', calculation_methods)

        # Apply custom CSS to make the select box smaller
        #st.markdown(
           # """
            #<style>
            #div[data-baseweb="select"] {
             #   max-width: 250px; /* Adjust the width as needed */
            #}
            #</style>
            #""",
            #unsafe_allow_html=True
        #)
        tab1, tab2, tab3, tab4 = st.tabs(["Today's Games", "Tomorrow's Games", "Injuries","Rankings and Awards 🏆"])

        with tab1:
                today_games['Projected_Score'] = (today_games['hometotal'] + today_games['vistotal']) 

                # Calculate the projected Money Line odds
                today_games['Projected_Line'] = 0.8 * today_games['ml1'] + 0.2 * today_games['ml2'] + 0 * today_games['ml3']

                # Round the constant to the nearest 0.5 using round_half_even
                today_games['Constant'] = np.round(today_games['Projected_Score'] / 0.5) * 0.5

                # Set the standard deviation
                std_deviation_overunder = 1.67
                std_deviation_ml = 2.48

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
                button_clicked = st.button("Generate Today's Odds")

                if button_clicked:
                    fanduel_nhl_odds = fetch_fanduel_nhl_odds()
                    
                    # Display the odds for today's games in a Streamlit table
                    st.write("### Today's Projected Odds:")
                    for i, game in enumerate(today_games.itertuples(), start=1):                   
                        st.subheader(f"{game.Visitor} *@* {game.Home}")
                        st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                        st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")
                        st.write(f"Projected Over Under Line: {game.Constant:.1f}")            
                        st.write(f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")
                                                    
                        if fanduel_nhl_odds:
                            st.write("NHL Moneyline Odds (FanDuel):")
                            for odds_game in fanduel_nhl_odds:
                                home_team = odds_game['home_team']
                                away_team = odds_game['away_team']
                                home_win_odds = odds_game['home_win_odds']
                                away_win_odds = odds_game['away_win_odds']
                                over_odds = odds_game['over_odds']
                                under_odds = odds_game['under_odds']
                                total_line = odds_game['total_line']
                                
                                # Check if the odds correspond to the current projected game
                                if home_team == game.Home and away_team == game.Visitor:
                                    st.write(f"{game.Home}: {home_win_odds}, {game.Visitor}: {away_win_odds}")
                                    st.write(f"Total Line: {total_line}")
                                    st.write(f"Over: {over_odds}, Under: {under_odds}")
                        else:
                            st.warning("No NHL odds found.")
                        # Display the expander button and its content

                        # Display the expander button and its content
                        with st.expander('More Details', expanded=False):
                            excel_file = 'nhl.xlsx'
                            excel_file2 = 'nhlgar.xlsx'
                            sheet_name = 'test'
                            sheet_name2 = 'Skater2024'
                                                                        
                            yesterday = datetime.now(time_zone).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                            yesterday_games = game_data[(game_data['Date'] >= yesterday) & (game_data['Date'] < yesterday + pd.DateOffset(1))]

                            home_yesterday = yesterday_games[(yesterday_games['Home'] == game.Home) | (yesterday_games['Visitor'] == game.Home)]
                            visitor_yesterday = yesterday_games[(yesterday_games['Home'] == game.Visitor) | (yesterday_games['Visitor'] == game.Visitor)]

                            last_matchup_date, last_matchup_scores = find_last_matchup(game.Home, game.Visitor, game_data, today)

                            if last_matchup_date:
                                if last_matchup_date <= today:  # Check if the last matchup date is in the past or today
                                    formatted_last_matchup_date = last_matchup_date.strftime('%Y-%m-%d')
                                    
                                    # Determine which score corresponds to which team
                                    home_score_index = 0 if game.Home == last_matchup_scores[0] else 1
                                    away_score_index = 1 - home_score_index
                                    
                                    home_score = last_matchup_scores[home_score_index]
                                    away_score = last_matchup_scores[away_score_index]
                                    
                                    st.write(f"Last matchup: {formatted_last_matchup_date}  -  {game.Visitor}: {away_score} vs {game.Home}: {home_score}")
                                else:
                                    st.write("No matchups this season.")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write(f"{game.Home}:")

                                    # Load the Excel file into a DataFrame
                                    df = pd.read_excel(excel_file2, sheet_name=sheet_name2)

                                    # Filter the DataFrame to get injured players
                                    injured_players = df[df['injury'] > 14]
                                  
                                                                     
                                    
                                    # Filter the DataFrame to get injuries for the specific team
                                    team_injuries = injured_players[injured_players['team1'] == game.Home]

                                    notable_injuries = team_injuries['Player'].tolist()
                                                              
                                    if notable_injuries:
                                        injuries_string = ", ".join(notable_injuries)
                                        st.write(f"Injuries:", injuries_string)
                                    else:
                                        st.write("No important injuries")
                                    
                                    # Check if the visitor team played yesterday
                                    if not home_yesterday.empty:
                                        st.write(f"{game.Home} played yesterday")
                                    else:
                                        pass

                                with col2:
                                    st.write(f"{game.Visitor}:")

                                    # Load the Excel file into a DataFrame
                                    df = pd.read_excel(excel_file2, sheet_name=sheet_name2)

                                    # Filter the DataFrame to get injured players
                                    injured_players = df[df['injury'] > 14]

                                    # Filter the DataFrame to get injuries for the specific team
                                    team_injuries = injured_players[injured_players['team1'] == game.Visitor]

                                    # Display notable injuries for the team
                                    notable_injuries = team_injuries['Player'].tolist()
                                    if notable_injuries:
                                        injuries_string = ", ".join(notable_injuries)
                                        st.write("Injuries:", injuries_string)
                                    else:
                                        st.write("No important injuries")
                                    
                                    # Check if the visitor team played yesterday
                                    if not visitor_yesterday.empty:
                                        st.write(f"{game.Visitor} played yesterday")
                                    else:
                                        pass


                    ##elif selected_method == 'American':
                        ##st.subheader('Coming Soon - Decimal Only')
            
            

        with tab2:                  
             
            ##if selected_method == 'Decimal':
                # Calculate and display the over/under odds, implied probabilities, and projected scores
                tomorrow_games['Projected_Score'] = (tomorrow_games['hometotal'] + tomorrow_games['vistotal']) 

                # Calculate the projected Money Line odds
                tomorrow_games['Projected_Line'] = 0.8 * tomorrow_games['ml1'] + 0.2 * tomorrow_games['ml2'] + 0 * tomorrow_games['ml3']

                # Round the constant to the nearest 0.5 using round_half_even
                tomorrow_games['Constant'] = np.round(tomorrow_games['Projected_Score'] / 0.5) * 0.5

                # Set the standard deviation
                std_deviation_overunder = 1.67
                std_deviation_ml = 2.48

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

                button_clicked = st.button("Generate Tomorrow's Odds")

        # Check if the button is clicked
                if button_clicked:
                    fanduel_nhl_odds = fetch_fanduel_nhl_odds()
                
                    for i, game in enumerate(tomorrow_games.itertuples(), start=1):
                        st.subheader(f"{game.Visitor} *@* {game.Home}")
                        st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                        st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")

                        st.write(f"Projected Over Under Line: {game.Constant:.1f}")
                        st.write(
                            f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")
                        if fanduel_nhl_odds:
                            st.write("NHL Moneyline Odds (FanDuel):")
                            for odds_game in fanduel_nhl_odds:
                                home_team = odds_game['home_team']
                                away_team = odds_game['away_team']
                                home_win_odds = odds_game['home_win_odds']
                                away_win_odds = odds_game['away_win_odds']
                                over_odds = odds_game['over_odds']
                                under_odds = odds_game['under_odds']
                                total_line = odds_game['total_line']
                                
                                # Check if the odds correspond to the current projected game
                                if home_team == game.Home and away_team == game.Visitor:
                                    st.write(f"{game.Home}: {home_win_odds}, {game.Visitor}: {away_win_odds}")
                                    st.write(f"Total Line: {total_line}")
                                    st.write(f"Over: {over_odds}, Under: {under_odds}")
                        else:
                            st.warning("Odds not released.")
                            # Display the expander button and its content
                        with st.expander('More Details', expanded=False):
                            excel_file = 'nhl.xlsx'
                            excel_file2 = 'nhlgar.xlsx'
                            sheet_name = 'test'
                            sheet_name2 = 'Skater2024'
                                                                        
                            yesterday = datetime.now(time_zone).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                            yesterday_games = game_data[(game_data['Date'] >= yesterday) & (game_data['Date'] < yesterday + pd.DateOffset(1))]
                            home_yesterday = yesterday_games[(yesterday_games['Home'] == game.Home) | (yesterday_games['Visitor'] == game.Home)]
                            visitor_yesterday = yesterday_games[(yesterday_games['Home'] == game.Visitor) | (yesterday_games['Visitor'] == game.Visitor)]

                            last_matchup_date, last_matchup_scores = find_last_matchup(game.Home, game.Visitor, game_data, today)

                            if last_matchup_date:
                                if last_matchup_date <= today:  # Check if the last matchup date is in the past or today
                                    formatted_last_matchup_date = last_matchup_date.strftime('%Y-%m-%d')
                                    
                                    # Determine which score corresponds to which team
                                    home_score_index = 0 if game.Home == last_matchup_scores[0] else 1
                                    away_score_index = 1 - home_score_index
                                    
                                    home_score = last_matchup_scores[home_score_index]
                                    away_score = last_matchup_scores[away_score_index]
                                    
                                    st.write(f"Last matchup: {formatted_last_matchup_date}  -  {game.Visitor}: {away_score} vs {game.Home}: {home_score}")
                                else:
                                    st.write("No matchups this season.")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write(f"{game.Home}:")

                                    # Load the Excel file into a DataFrame
                                    df = pd.read_excel(excel_file2, sheet_name=sheet_name2)

                                    # Filter the DataFrame to get injured players
                                    injured_players = df[df['injury'] > 14]
                                  
                                                                     
                                    
                                    # Filter the DataFrame to get injuries for the specific team
                                    team_injuries = injured_players[injured_players['team1'] == game.Home]

                                    notable_injuries = team_injuries['Player'].tolist()
                                                              
                                    if notable_injuries:
                                        injuries_string = ", ".join(notable_injuries)
                                        st.write(f"Injuries:", injuries_string)
                                    else:
                                        st.write("No important injuries")
                                    
                                   
                                    if game.Home in today_games['Home'].values or game.Home in today_games['Visitor'].values:
                                        st.write(f"{game.Home} back-to-back game")
                                    else :
                                        pass
                    

                                with col2:
                                    st.write(f"{game.Visitor}:")

                                    # Load the Excel file into a DataFrame
                                    df = pd.read_excel(excel_file2, sheet_name=sheet_name2)

                                    # Filter the DataFrame to get injured players
                                    injured_players = df[df['injury'] > 14]

                                    # Filter the DataFrame to get injuries for the specific team
                                    team_injuries = injured_players[injured_players['team1'] == game.Visitor]

                                    # Display notable injuries for the team
                                    notable_injuries = team_injuries['Player'].tolist()
                                    if notable_injuries:
                                        injuries_string = ", ".join(notable_injuries)
                                        st.write("Injuries:", injuries_string)
                                    else:
                                        st.write("No important injuries")
                                    
                                    # Check if the visitor team played yesterday
                                    if game.Visitor in today_games['Home'].values or game.Visitor in today_games['Visitor'].values:
                                        st.write(f"{game.Visitor} back-to-back game")
                                    else:
                                        pass
                
            ##elif selected_method == 'American':
                ##st.subheader('Coming soon - Decimal only')
                    
        with tab3:
                excel_file = 'nhlgar.xlsx'
                sheet_name = 'Injuries'
                st.subheader('NHL Injuries 🚫')
                # Load data from the specified sheet in the first Excel file
                injury_data = pd.read_excel(excel_file, sheet_name=sheet_name)
                teams = [
                    "Anaheim Ducks", "Arizona Coyotes", "Boston Bruins", "Buffalo Sabres", "Calgary Flames",
                    "Carolina Hurricanes", "Chicago Blackhawks", "Colorado Avalanche", "Columbus Blue Jackets",
                    "Dallas Stars", "Detroit Red Wings", "Edmonton Oilers", "Florida Panthers", "Los Angeles Kings",
                    "Minnesota Wild", "Montreal Canadiens", "Nashville Predators", "New Jersey Devils",
                    "New York Islanders", "New York Rangers", "Ottawa Senators", "Philadelphia Flyers",
                    "Pittsburgh Penguins", "San Jose Sharks", "Seattle Kraken", "St. Louis Blues", "Tampa Bay Lightning",
                    "Toronto Maple Leafs", "Vancouver Canucks", "Vegas Golden Knights", "Washington Capitals", "Winnipeg Jets"
                ]


                # Create a selection box for choosing the team
                selected_team = st.selectbox('Select Team:', teams)


                # Filter the injury data based on the selected team
                filtered_data = injury_data[injury_data['Team'] == selected_team]
                filtered_data.rename(columns={'Injury Note': 'Injury'}, inplace=True)

                # Select columns to display
                columns_to_display = ['Player', 'Injury']


                # Convert DataFrame to HTML table without index
                html_table = filtered_data[columns_to_display].to_html(index=False)

                # Add CSS styling to center the headers
                html_table = html_table.replace('<thead>', '<thead style="text-align: center;"><style> th { text-align: center; }</style>', 1)

                # Display the HTML table in Streamlit
                st.write(html_table, unsafe_allow_html=True)

        with tab4:
            # Assuming 'Power Rankings' sheet contains the data
            excel_file = 'nhl.xlsx'
            excel_file2 = 'nhlgar.xlsx'
            sheet_name = 'Power Rankings'
            sheet_name2 = "Playerrankings"
            

            # Load data from the specified sheet in the first Excel file
            game_data = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Sort the data based on the 'powerranking' column (assuming it's in column 'powerranking')
            sorted_data = game_data.sort_values(by='powerranking')
            
            # Load data from the specified sheet in the second Excel file
            game_data2 = pd.read_excel(excel_file2, sheet_name=sheet_name2)

            # Sort the data based on the 'topplayer' column (assuming it's in column 'topplayer')
            sorted_data2 = game_data2.sort_values(by='Rank')
            
            # Create a two-column layout
            col1, col2 = st.columns(2)  

        
                # Display content in the left column
            with col1:
                        
                st.subheader("Team Rankings")

            # Rename the columns for display
                display_data = sorted_data[['powerranking', 'Team']].rename(columns={'Team': 'Team', 'powerranking': 'Rank'})

                # Display the DataFrame without the index column
                st.dataframe(display_data, hide_index=True)

                st.subheader("Model's Top Players")

                # Rename the columns for display and round 'mpg' to one decimal place
                display_data = sorted_data2[['Rank','topplayer', 'playteam','pos', 'gp', 'g', 'p', 'mpg']].rename(columns={'Rank': 'Rank','topplayer': 'Player', 'playteam': 'Team','pos': 'Pos', 'gp': 'GP', 'g':'Goals','p':'Points', 'mpg': 'MPG'})

                # Handle non-finite values before rounding and converting to integers
                display_data['MPG'] = display_data['MPG'].apply(lambda x: f"{x:.1f}" if not pd.isnull(x) else "0.000")

                display_data['Points'] = display_data['Points'].fillna(0).astype(int)
                display_data['Goals'] = display_data['Goals'].fillna(0).astype(int)
                display_data['GP'] = display_data['GP'].fillna(0).astype(int)

                # Display the DataFrame without the index column
                st.dataframe(display_data.head(15), hide_index=True)
            # Add your code for the right column here
            with col2:

                st.subheader("Model's Top Goalies")

                display_data = sorted_data2[['Rank', 'topgoalie', 'golteam', 'gs', 'sv%', 'qs']].rename(columns={'topgoalie': 'Goalie', 'golteam': 'Team', 'gs': 'GS', 'sv%': 'SV%', 'qs': 'Quality Starts'})

                # Handle non-finite values before converting to integers
                display_data['GS'] = display_data['GS'].fillna(0).astype(int)
                display_data['Quality Starts'] = display_data['Quality Starts'].fillna(0).astype(int)
                
                # Display the DataFrame without the index column
                st.dataframe(display_data.head(10), hide_index=True)

                    
            
                st.subheader("Model's Top Rookies")
                display_data = sorted_data2[['Rank','bestrookies', 'rook team', 'pos1', 'gp1', 'g1','p1','mpg1']].rename(columns={'Rank': 'Rank','bestrookies': 'Rookie', 'rook team': 'Team', 'gp1': 'GP', 'pos1': 'Pos', 'mpg1': 'MPG','g1':'Goals','p1':'Points'})

                # Handle non-finite values before rounding and converting to integers
                display_data['MPG'] = display_data['MPG'].apply(lambda x: f"{x:.1f}" if not pd.isnull(x) else "0.000")
                display_data['Points'] = display_data['Points'].fillna(0).astype(int)
                display_data['Goals'] = display_data['Goals'].fillna(0).astype(int)
                display_data['GP'] = display_data['GP'].fillna(0).astype(int)
                    
                # Display the DataFrame without the index column
                st.dataframe(display_data, hide_index=True)        

 
                    
elif selection == '🏀 NBA Model':
    st.title('NBA Model 🏀')
    st.header("How the Model Works")
    st.write("The model generates odds from its projected probability of outcomes. Think of these odds as the minimum return you would require to make a positive EV bet.")
    st.write("If a player is listed on the injury report as day-to-day the model will include them. The injury tab shows all injuries with details.")                               
    # Use a relative path to the Excel file
    excel_file = 'nba.xlsm'

    # Load data from "Game Data" sheet
    game_data = pd.read_excel(excel_file, sheet_name="2024schedule")

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
    def find_last_matchup(team1, team2, data, today):
        yesterday = today - timedelta(days=1)
        last_matchup_date = None
        last_matchup_scores = None
        
        # Check if there are any previous matchups between the teams
        matchups = ((data['Visitor'] == team1) & (data['Home'] == team2) | (data['Visitor'] == team2) & (data['Home'] == team1))
        if matchups.any():
            # Iterate over the DataFrame in reverse chronological order
            for index, row in data.loc[matchups].sort_values(by='Date', ascending=False).iterrows():
                # Exclude today's games
                if row['Date'] != today:
                    last_matchup_date = row['Date']
                    if row['Visitor'] == team1:
                        last_matchup_scores = (row['PTS1'], row['PTS'])
                    else:
                        last_matchup_scores = (row['PTS'], row['PTS1'])
                    if last_matchup_date <= yesterday:  # Check if the last matchup date is on or before yesterday
                        break  # Break out of the loop after finding the last matchup

        return last_matchup_date, last_matchup_scores
    def get_team_id(team_name):
        team_id_mapping = {}
        with open('nbateamid.txt', 'r') as f:
            for line in f:
                team_name_from_file, team_id = line.strip().split(',')  
                team_id_mapping[team_name_from_file] = int(team_id) # Convert ID to integer

        return team_id_mapping.get(team_name)
    
    # Customize for the current season 
    current_season = "2023-24"  

    # Get standings via the API
    standings_data = leaguestandingsv3.LeagueStandingsV3(
    )

    # Get the DataFrame directly
    standings_df = standings_data.get_data_frames()[0]
    standings_df['TeamCity'] = standings_df['TeamCity'].replace('LA', 'Los Angeles')


    # Select columns, sort, and perform any other manipulations as needed
    columns_to_keep = ['TeamName','TeamCity','L10', 'Last10Home', 'Last10Road','WinPCT'] 
    standings_df = standings_df[columns_to_keep]\
    
    




    

    tab1, tab2, tab3, tab4 = st.tabs(["Today's Games", "Tomorrow's Games", "Injuries","Rankings and Awards 🏆"])

    with tab1:
            button_clicked = st.button("Generate Today's Odds")

# Check if the button is clicked
            if button_clicked:

        
                @st.cache_data
                def skipComputation(today_games):
                    # Calculate the projected Money Line odds
                    today_games['Projected_Line'] = 0.5 * today_games['ml1'] + 0.5 * today_games['ml2'] + 0 * today_games['ml3']
                    today_games['Projected_Score'] = 1 * (today_games['homtot'] + today_games['vistot']) + 0 * (today_games['homtot1'] + today_games['vistot1'])   
                    today_games['Constant'] = np.round(today_games['Projected_Score'] / 0.5) * 0.5

                    # Set the standard deviation
                    std_deviation_overunder = 11.1
                    std_deviation_ml = 10.5

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

                    return today_games

                # Call the function to compute the values
                today_games = skipComputation(today_games)
                # Calculate last 10 win percentage


                st.write("### Today's Projected Odds:")
                for i, game in enumerate(today_games.itertuples(), start=1):
                    st.subheader(f"{game.Visitor} *@* {game.Home}")
                    st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                    st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")
                    st.write(f"Projected Over Under Line: {game.Constant:.1f}")            
                    st.write(f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")
            
            
                    # Display the expander button and its content
                    with st.expander('More Details', expanded=False):
            
                    ##if st.button(f"More Info: {game.Home} vs {game.Visitor}"):
                        excel_file = 'nba.xlsm'
                        sheet_name = '2024EPM'
                        sheet_name1 = '2024schedule'
                        
                        yesterday = datetime.now(time_zone).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                        yesterday_games = game_data[(game_data['Date'] >= yesterday) & (game_data['Date'] < yesterday + pd.DateOffset(1))]

                        home_yesterday = yesterday_games[(yesterday_games['Home'] == game.Home) | (yesterday_games['Visitor'] == game.Home)]
                        visitor_yesterday = yesterday_games[(yesterday_games['Home'] == game.Visitor) | (yesterday_games['Visitor'] == game.Visitor)]

                        last_matchup_date, last_matchup_scores = find_last_matchup(game.Home, game.Visitor, game_data, today)

                        if last_matchup_date:
                            if last_matchup_date <= today:  # Check if the last matchup date is in the past or today
                                formatted_last_matchup_date = last_matchup_date.strftime('%Y-%m-%d')
                                
                                # Determine which score corresponds to which team
                                home_score_index = 0 if game.Home == last_matchup_scores[0] else 1
                                away_score_index = 1 - home_score_index
                                
                                home_score = last_matchup_scores[home_score_index]
                                away_score = last_matchup_scores[away_score_index]
                                
                                st.write(f"Last matchup: {formatted_last_matchup_date}  -  {game.Visitor}: {away_score} vs {game.Home}: {home_score}")
                            else:
                                st.write("No matchups this season.")
                            # Create a two-column layout
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"{game.Home}:")

                        
                            # Load the Excel file into a DataFrame
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)

                            # Filter the DataFrame to get injured players
                            injured_players = df[df['missing'] > 20]

                            # Filter the DataFrame to get injuries for the specific team
                            team_injuries = injured_players[injured_players['team1'] == game.Home]

                            # Display notable injuries for the team
                            notable_injuries = team_injuries['name'].tolist()
                            if notable_injuries:
                                injuries_string = ", ".join(notable_injuries)
                                st.write(f" * Injuries:", injuries_string)
                            else:
                                st.write(f" * No important injuries")
                            # Check if the visitor team played yesterday

                            standings_df['Combined'] = standings_df['TeamCity'] + " " + standings_df['TeamName']
                            
                            home_team_data = standings_df[standings_df['Combined'] == game.Home]  

                            l10_home = home_team_data['L10'].iloc[0]
                            last10home_home = home_team_data['Last10Home'].iloc[0]
                            last10road_home = home_team_data['Last10Road'].iloc[0]

                            # Calculate last 10 win percentage 
                            wins_last_10 = int(l10_home.split('-')[0])  
                            last_10_win_pct = wins_last_10 / 10   

                            # Get overall win percentage
                            overall_win_pct = home_team_data['WinPCT'].iloc[0]

                            # Determine emoji with threshold
                            threshold = 0.2
                            if last_10_win_pct >= overall_win_pct + threshold: 
                                emoji = "🔥" 
                            elif last_10_win_pct <= overall_win_pct - threshold:
                                emoji = "🥶"  # Cold emoji 
                            else:
                                emoji = "➖"  # No emoji if within the threshold

                            # Display with emoji
                            st.write(f" * Last 10: {l10_home} / Last 10 Home: {last10road_home} / Recent Trend: {emoji}") 
                              
                            if not home_yesterday.empty:
                                st.write(f" * {game.Home} played yesterday")
                            else:
                                pass
                            

                        with col2:
                            st.write(f"{game.Visitor}:")

                                        

                            # Load the Excel file into a DataFrame
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)

                            # Filter the DataFrame to get injured players
                            injured_players = df[df['missing'] > 20]

                            # Filter the DataFrame to get injuries for the specific team
                            team_injuries = injured_players[injured_players['team1'] == game.Visitor]

                            # Display notable injuries for the team
                            notable_injuries = team_injuries['name'].tolist()
                            if notable_injuries:
                                injuries_string = ", ".join(notable_injuries)
                                st.write(f" * Injuries:", injuries_string)
                            else:
                                st.write(f" * No important injuries")
                            # Check if the visitor team played yesterday

                            standings_df['Combined'] = standings_df['TeamCity'] + " " + standings_df['TeamName']

                            vis_team_data = standings_df[standings_df['Combined'] == game.Visitor]

                            l10_vis = vis_team_data['L10'].iloc[0]
                            last10road_vis = vis_team_data['Last10Road'].iloc[0]

                            # Calculate last 10 win percentage for the visitor
                            wins_last_10_vis = int(l10_vis.split('-')[0])  
                            last_10_win_pct_vis = wins_last_10_vis / 10   

                            # Get overall win percentage for the visitor
                            overall_win_pct_vis = vis_team_data['WinPCT'].iloc[0]

                            # Determine emoji with threshold for the visitor
                            threshold = 0.2
                            if last_10_win_pct_vis >= overall_win_pct_vis + threshold: 
                                emoji_vis = "🔥" 
                            elif last_10_win_pct_vis <= overall_win_pct_vis - threshold:
                                emoji_vis = "🥶"  # Cold emoji 
                            else:
                                emoji_vis = "➖" 

                            # Display stats for the visitor 
                            st.write(f" * Last 10: {l10_vis} / Last 10 Away: {last10road_vis} / Recent Trend: {emoji_vis}") 
                            
                            if not visitor_yesterday.empty:
                                st.write(f" * {game.Visitor} played yesterday")
                            else:
                                pass

                    



              
    with tab2:
        button_clicked = st.button("Generate Tomorrow's Odds")

# Check if the button is clicked
        if button_clicked:
            # Calculate and display the over/under odds, implied probabilities, and projected scores based on the selected method
            # Calculate the projected Money Line odds
            tomorrow_games['Projected_Line'] = 0.4 * tomorrow_games['ml1'] + 0.6 * tomorrow_games['ml2'] + 0 * tomorrow_games['ml3']

            tomorrow_games['Projected_Score'] = 1 * (tomorrow_games['homtot'] + tomorrow_games['vistot']) + 0 * (tomorrow_games['homtot1'] + tomorrow_games['vistot1'])

            # Round the constant to the nearest 0.5 using round_half_even
            tomorrow_games['Constant'] = np.round(tomorrow_games['Projected_Score'] / 0.5) * 0.5

            # Set the standard deviation
            std_deviation_overunder = 11.1
            std_deviation_ml = 10.5

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
            # Display tomorrow's games and projected odds
            st.write("### Tomorrow's Projected Odds:")
            for i, game in enumerate(tomorrow_games.itertuples(), start=1):
                st.subheader(f"{game.Visitor} *@* {game.Home}")
                st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")
                st.write(f"Projected Over Under Line: {game.Constant:.1f}")            
                st.write(f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")
                
                # Create an expander for more details of each game
                with st.expander(f"More Details for Game {i}", expanded=False):
                    excel_file = 'nba.xlsm'
                    sheet_name = '2024EPM'
                    
                    # Check if either team played yesterday
                    yesterday = datetime.now(time_zone).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                    yesterday_games = game_data[(game_data['Date'] >= yesterday) & (game_data['Date'] < yesterday + pd.DateOffset(1))]
                    home_yesterday = yesterday_games[(yesterday_games['Home'] == game.Home) | (yesterday_games['Visitor'] == game.Home)]
                    visitor_yesterday = yesterday_games[(yesterday_games['Home'] == game.Visitor) | (yesterday_games['Visitor'] == game.Visitor)]
                    
                    # Assuming find_last_matchup() returns last_matchup_date and last_matchup_scores as a tuple
                    last_matchup_date, last_matchup_scores = find_last_matchup(game.Home, game.Visitor, game_data, today)

                    if last_matchup_date:
                        if last_matchup_date <= today:  # Check if the last matchup date is in the past or today
                            formatted_last_matchup_date = last_matchup_date.strftime('%Y-%m-%d')
                            
                            # Determine which score corresponds to which team
                            home_score_index = 0 if game.Home == last_matchup_scores[0] else 1
                            away_score_index = 1 - home_score_index
                            
                            home_score = last_matchup_scores[home_score_index]
                            away_score = last_matchup_scores[away_score_index]
                            
                            st.write(f"Last matchup: {formatted_last_matchup_date}  -  {game.Visitor}: {away_score} vs {game.Home}: {home_score}")
                        else:
                            st.write("No matchups this season.")


                    
                    # Create a two-column layout for displaying team injuries

                    col1, col2 = st.columns(2)
                    with col1:
                            st.subheader(f"{game.Home}:")
                            # Load the Excel file into a DataFrame
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)
                            # Filter the DataFrame to get injured players
                            injured_players = df[df['missing'] > 20]
                            # Filter the DataFrame to get injuries for the specific team
                            team_injuries = injured_players[injured_players['team1'] == game.Home]
                            # Display notable injuries for the team
                            notable_injuries = team_injuries['name'].tolist()
                            if notable_injuries:
                                injuries_string = ", ".join(notable_injuries)
                                st.write(f"Injuries:", injuries_string)
                            else:
                                st.write("No important injuries")

                            standings_df['Combined'] = standings_df['TeamCity'] + " " + standings_df['TeamName']
                            home_team_data = standings_df[standings_df['Combined'] == game.Home]  
                            
                            l10_home = home_team_data['L10'].iloc[0]
                            last10home_home = home_team_data['Last10Home'].iloc[0]
                            last10road_home = home_team_data['Last10Road'].iloc[0]

                                    # Calculate last 10 win percentage 
                            wins_last_10 = int(l10_home.split('-')[0])  
                            last_10_win_pct = wins_last_10 / 10   

                                    # Get overall win percentage
                            overall_win_pct = home_team_data['WinPCT'].iloc[0]

                                    # Determine emoji with threshold
                            threshold = 0.2
                            if last_10_win_pct >= overall_win_pct + threshold: 
                                        emoji = "🔥" 
                            elif last_10_win_pct <= overall_win_pct - threshold:
                                        emoji = "🥶"  # Cold emoji 
                            else:
                                        emoji = "➖"  # No emoji if within the threshold

                                    # Display with emoji
                            st.write(f" * Last 10: {l10_home} / Last 10 Home: {last10road_home} / Recent Trend: {emoji}")  
                                    
                                # Check if the home team played yesterday
                            if game.Home in today_games['Home'].values or game.Home in today_games['Visitor'].values:
                                st.write(f"{game.Home} back-to-back game")
                            else :
                                pass
                    
                    with col2:
                            st.subheader(f"{game.Visitor}:")
                            # Load the Excel file into a DataFrame
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)
                            # Filter the DataFrame to get injured players
                            injured_players = df[df['missing'] > 20]
                            # Filter the DataFrame to get injuries for the specific team
                            team_injuries = injured_players[injured_players['team1'] == game.Visitor]
                            # Display notable injuries for the team
                            notable_injuries = team_injuries['name'].tolist()
                            if notable_injuries:
                                injuries_string = ", ".join(notable_injuries)
                                st.write("Injuries:", injuries_string)
                            else:
                                st.write("No important injuries")
                                                      
                            vis_team_data = standings_df[standings_df['Combined'] == game.Visitor]

                            
                            l10_vis = vis_team_data['L10'].iloc[0]
                            last10road_vis = vis_team_data['Last10Road'].iloc[0]

                                # Calculate last 10 win percentage for the visitor
                            wins_last_10_vis = int(l10_vis.split('-')[0])  
                            last_10_win_pct_vis = wins_last_10_vis / 10   

                                # Get overall win percentage for the visitor
                            overall_win_pct_vis = vis_team_data['WinPCT'].iloc[0]

                                # Determine emoji with threshold for the visitor
                            threshold = 0.2
                            if last_10_win_pct_vis >= overall_win_pct_vis + threshold: 
                                    emoji_vis = "🔥" 
                            elif last_10_win_pct_vis <= overall_win_pct_vis - threshold:
                                    emoji_vis = "🥶"  # Cold emoji 
                            else:
                                    emoji_vis = "➖" 

                                # Display stats for the visitor 
                            st.write(f" * Last 10: {l10_vis} / Last 10 Away: {last10road_vis} / Recent Trend: {emoji_vis}") 

                       

                            # Check if the visitor team played yesterday
                            if game.Visitor in today_games['Home'].values or game.Visitor in today_games['Visitor'].values:
                                st.write(f"{game.Visitor} back-to-back game")
                            else:
                                pass
                                                    



    with tab3:
        excel_file = 'nba.xlsm'
        sheet_name = 'Injuries'
        st.title('NBA Injuries 🚫')
            # Load data from the specified sheet in the first Excel file
        injury_data = pd.read_excel(excel_file, sheet_name=sheet_name)
        teams = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
            "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
            "Utah Jazz", "Washington Wizards"
        ]

        # Create a selection box for choosing the team
        selected_team = st.selectbox('Select Team:', teams)

        # Filter the injury data based on the selected team
        filtered_data = injury_data[injury_data['Team'] == selected_team]
        filtered_data.rename(columns={'Description': 'Injury'}, inplace=True)

        # Select columns to display
        columns_to_display = ['Player', 'Injury']


        # Convert DataFrame to HTML table without index
        html_table = filtered_data[columns_to_display].to_html(index=False)

        # Add CSS styling to center the headers
        html_table = html_table.replace('<thead>', '<thead style="text-align: center;"><style> th { text-align: center; }</style>', 1)

        # Display the HTML table in Streamlit
        st.write(html_table, unsafe_allow_html=True)
    
    with tab4:
        excel_file = 'nba.xlsm'
        sheet_name = 'Powerrankings'
        sheet_name2 = "Topplayers"
    # Load data from the specified sheet in the first Excel file
        game_data = pd.read_excel(excel_file, sheet_name=sheet_name)
        game_data2 = pd.read_excel(excel_file, sheet_name=sheet_name2)

        # Sort the data (Do this once before your `display_...` functions)
        sorted_data = game_data.sort_values(by='Power') 
        sorted_data2 = game_data2.sort_values(by='Rank')
        def display_team_rankings(data):
            st.subheader("Power Rankings")
            display_data = data[['Power', 'Team','Off Rank','Def Rank']].rename(columns={'Power': 'Ranking'})
            st.table(display_data)  # No need for hide_index=True (improved in newer Streamlit)

        def display_top_players(data):
            st.subheader("Model's Top Players")
            display_data = data[['Rank', 'topplayer', 'playteam', 'PTS', 'AST', 'REB','STL','BLK']].rename(columns={'topplayer': 'Player', 'playteam': 'Team'})
            st.table(display_data.head(15))

        def display_top_rookies(data):
            st.subheader("Models Top Rookies")
            display_data = data[['Rank', 'bestrook', 'rookteam', 'p1', 'a1', 'r1','s1','b1']].rename(columns={'bestrook': 'Player', 'rookteam': 'Team', 'p1': 'PTS', 'a1': 'AST','r1': 'REB', 's1': 'STL', 'b1' : 'BLK'})
            st.table(display_data.head(10))

    
        pages  = [
        "Power Rankings","MVP Race", "Rookie of the Year","Most Improved Player"
        ]
        
        # Create a selection box for choosing the team
        selected_team = st.selectbox('Select Power Ranking:', pages)

        # Display tables based on selection
        if selected_team == "Power Rankings":
            display_team_rankings(sorted_data) 
        elif selected_team == "MVP Race":
            display_top_players(sorted_data2)  # Assuming MVP data is in sorted_data2
        elif selected_team == "Rookie of the Year":
            display_top_rookies(sorted_data2)  # Assuming Rookie data is in sorted_data2
        elif selected_team == "Most Improved Player":
            # Add a placeholder or a message stating this feature is not yet implemented
            st.write("Most Improved Player rankings coming soon!") 





elif selection == '🏀 NBA Franchise Mode':
    excel_file = 'nba.xlsm'
    st.title('Welcome to Franchise Mode')
    st.write('Filter NBA contracts and value players franchise worth.')

    tab1, tab2 = st.tabs(["Team Totals", "Player Value"])

    with tab1:
        sheet_name1 = 'franstan'         
        franchise_data1 = pd.read_excel(excel_file, sheet_name=sheet_name1)
        sorted_data2 = franchise_data1.sort_values(by='salRank', ascending= False)
        sorted_data2.rename(columns={'salRank': 'Franchise Score', 'Draftcap' : 'Draft Capital Rank'}, inplace=True)
        sorted_data2['Franchise Score'] = sorted_data2['Franchise Score'].round(1)

        # Select columns to display
        columns_to_display = ['Team','Franchise Score', 'Draft Capital Rank']

        # Convert DataFrame to HTML table without index
        html_table = sorted_data2[columns_to_display].to_html(index=False)

        # Add CSS styling to center the headers 
        html_table = html_table.replace('<thead>', '<thead style="text-align: center;"><style> th { text-align: center; }</style>', 1)

        # Display the HTML table in Streamlit
        st.write(html_table, unsafe_allow_html=True)
        
    with tab2:
        excel_file = 'nba.xlsm'
        sheet_name = 'fran'
        
        
        franchise_data = pd.read_excel(excel_file, sheet_name=sheet_name)
       
        teams = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
            "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
            "Utah Jazz", "Washington Wizards"
        ]

        # Create a selection box for choosing the team
        selected_team = st.selectbox('Select Team:', teams)
        # Filter the data based on the selected team
        filtered_data = franchise_data[franchise_data['team'] == selected_team]

        # Sort the filtered data by the 'Value' column
        filtered_data = filtered_data.sort_values(by='Value', ascending=False)
        
        # Format 'AAV' and 'Age' columns to display with two decimal points and no decimal points respectively
        filtered_data['AAV'] = filtered_data['AAV'].round(2)
        filtered_data['Age'] = filtered_data['Age'].round(0)

        # Format 'Value' column to display with two decimal points
        filtered_data['Value'] = filtered_data['Value'].apply(lambda x: f'{x:.2f}')

        # Select columns to display
        columns_to_display = ['Player', 'Age', 'AAV', 'Years', 'Value']

        # Convert DataFrame to HTML table without index
        html_table = filtered_data[columns_to_display].to_html(index=False)

        # Add CSS styling to center the headers
        html_table = html_table.replace('<thead>', '<thead style="text-align: center;"><style> th { text-align: center; }</style>', 1)

        # Display the HTML table in Streamlit
        st.write(html_table, unsafe_allow_html=True)



elif selection == '🔑 Betting Strategy':
    tab1, tab2, tab3= st.tabs(["Overview", "Expected Value", "Varience"])
    with tab1:
            st.title('Strategy 🔑')
            st.write('Our goal at Quantum Odds is to make our clients money. To make money gambling, you need relentless discipline and a strategy with an edge. The sports betting market is generally efficient and set up for you to lose (like all gambling). Lines are priced to the sportsbook’s implied probabilities, making every bet have an expected return of zero. After rake, every bet has negative expected return, explaining why most people lose. However, bettors have one strategic advantage over sportsbooks: flexibility. Sportsbooks need to post lines every day for every game in every sport and will eventually post a non-competitive line. Profitable bettors shop the market for these mistakes. Below is a step-by-step guide for how Quantum Odds will help you generate an edge.')            
            st.subheader('Data Models')
            st.write('The first step to our strategy is to generate odds before they are released. Using statistics and machine learning, we crunch real time player and team data to produce our lines. Our data models produce lines the same way sportsbooks do. Having lines ready before the sportsbook is crucial to allowing you to find inefficiencies quickly.')
            st.subheader('Speed and Sportsbook Selection')   
            st.write('Making positive EV bets is 75% data and 25% speed. The best time to find a great bet is right after a sportsbook drops their lines. After release, lines will shift rapidly as sharks place large bets on inefficiencies. Sportsbooks know these bets are sharp and will adjust their lines appropriately making them stronger. Compare Sportsbooks and find which ones consistently release odds first. Track when lines are released and place your bets within an hour of them being released. When the odds are dropped compare them to Quantum Odd’s models and find the inefficiencies.')
            st.subheader('Line Shopping')
            st.write('The last step of maximizing your EV is to line shop your bets. Place all your bets on the book that releases lines first, then watch other books release their odds. Generally, the closer to gametime the sharper the odds but sometimes you can cash out on your original bet to get better odds at another book. Do note, it rarely works to line shop if you pay a cash out penalty and this should be considered when picking your main book.')
            st.write('Line shopping also allows you to back-test your bets. If lines consistently move in your favor, you must be making high value bets. ') 
            st.subheader('Discipline') 
            st.write('1. Unit size 2-4% of bank roll per bet')
            st.write('2. Don’t bet for the sake of betting - no inefficiencies no bet')
            st.write('3. Understand variance and how it impacts your returns')
    with tab2:
        st.title('What is Expected Value')

    with tab3:
        st.title('Varience in Gambling')

elif selection == '💲Performance Tracking':
    # Define functions
    def get_current_value(file_path, column_name):
        df = pd.read_excel(file_path)
        last_value = df[column_name].iloc[-1]
        return last_value

    def calculate_win_loss_push(column):
        wins = int(column.str.count('y').sum())
        losses = int(column.str.count('n').sum())
        pushes = int(column.str.count('p').sum())
        return f'{wins} Wins - {losses} Losses - {pushes} Push'

    def calculate_percentage_return(starting_bank_role, current_bank_role):
        return (current_bank_role / starting_bank_role - 1) * 100

    
    st.title('Performance Tracking')

    
    file_path = 'Performance.xlsx'  

    # Column names
    column_name_balance = 'Balance'  # Replace with the actual name of your balance column
    column_name_outcomes = 'Win'  # Replace with the actual name of your outcomes column

    # Selection logic
    if selection == '💲Performance Tracking':
        # Get the current bank role
        current_bank_role = get_current_value(file_path, column_name_balance)

        # Read Excel file for outcomes
        df_outcomes = pd.read_excel(file_path)

        # Extract the specified column for outcomes
        outcomes_column = df_outcomes[column_name_outcomes]

        # Call the function with the correct column
        result = calculate_win_loss_push(outcomes_column)

        # Calculate percentage return
        starting_bank_role = 250  # Assuming 'starting_bank_role' is 250 (as mentioned in your code)
        percentage_return = calculate_percentage_return(starting_bank_role, current_bank_role)

        # Set color based on the sign of percentage return
        color = 'green' if percentage_return >= 0 else 'red'

        # Display current record and percentage return
        st.subheader(result)
        st.write(f'Starting Bank Roll = {starting_bank_role} | Current Bank Roll = {current_bank_role:.2f}')

    
        # Format the percentage return for display
        formatted_percentage_return = f'<span style="font-size:24px; color:{color}; font-weight:bold;">{percentage_return:.2f}%</span>'
        # Use st.markdown for displaying HTML content
        st.markdown(f'Percentage Return: {formatted_percentage_return}', unsafe_allow_html=True)

        # Create line chart
        fig = px.line(df_outcomes, x='Date', y=['Starting Balance', 'Current Balance'], labels={'value': 'Balance'})

        # Convert 'Date' column to string
        df_outcomes['Date'] = df_outcomes['Date'].astype(str)

        # Display the chart using st.plotly_chart
        st.plotly_chart(fig, use_container_width=True)


   

