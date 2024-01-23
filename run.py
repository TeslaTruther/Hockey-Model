import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from PIL import Image
import os
import scipy.stats as stats
import plotly.express as px

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
selection = st.sidebar.radio('Quantum Odds 	✅', ['🏠 Home', '🏒 NHL Model', '📊 NHL Power Rankings', '🚫 NHL Injuries', '🏀 NBA Model','📊 NBA Power Rankings', '🚫 NBA Injuries','🔑 Betting Strategy', '💲Performance Tracking'])

if selection == '🏠 Home':
    # Main content
    st.title("Quantum Odds 	✅")
    st.write("We generate odds so you can compete against sportsbooks.")
    st.write("Find inefficient markets and make positive expected value bets.")   
     
    st.image(resized_pandas[0])    
    # Add a URL at the bottom
    st.markdown('Share this site: '"https://quantumodds.streamlit.app/")  
elif selection == '🏒 NHL Model':
    
    if selection == '🏒 NHL Model':
       
       
        excel_file = 'nhl.xlsx'

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
        st.title('NHL Model 🏒 ')
        st.header("How the Model Works")
        st.write("The model generates odds from its projected probability of outcomes. Think of these odds as the minimum return you would require to make a positive EV bet.")
        st.subheader("Run The Model:")

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
    
    # Button to get today's odds
        if st.button("Generate Today's Odds", key="get_today_odds"):
            # Calculate and display the over/under odds, implied probabilities, and projected scores based on the selected method
            ##if selected_method == 'Decimal':
                # Calculate and display the over/under odds, implied probabilities, and projected scores
                today_games['Projected_Score'] = (today_games['hometotal'] + today_games['vistotal']) 

                # Calculate the projected Money Line odds
                today_games['Projected_Line'] = 0.3 * today_games['ml1'] + 0.45 * today_games['ml2'] + 0.25 * today_games['ml3']

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

                # Display the odds for today's games in a Streamlit table
                st.write("### Today's Games and Projected Odds:")
                for i, game in enumerate(today_games.itertuples(), start=1):                   
                    st.subheader(f"{game.Visitor} *@* {game.Home}")
                    st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                    st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")

                    st.write(f"Projected Over Under Line: {game.Constant:.1f}")            
                    st.write(f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")

            ##elif selected_method == 'American':
                ##st.subheader('Coming Soon - Decimal Only')


          

                    
        if st.button("Generate Tomorrow's Odds", key="get_tomorrows_odds"):
        
            ##if selected_method == 'Decimal':
                # Calculate and display the over/under odds, implied probabilities, and projected scores
                tomorrow_games['Projected_Score'] = (tomorrow_games['hometotal'] + tomorrow_games['vistotal']) 

                # Calculate the projected Money Line odds
                tomorrow_games['Projected_Line'] = 0.3 * tomorrow_games['ml1'] + 0.45 * tomorrow_games['ml2'] + 0.25 * tomorrow_games['ml3']

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

                # Display the odds for tomorrow's games in a Streamlit table
                st.write("### Tomorrow's Games and Projected Odds:")

                for i, game in enumerate(tomorrow_games.itertuples(), start=1):
                    st.subheader(f"{game.Visitor} *@* {game.Home}")
                    st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                    st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")

                    st.write(f"Projected Over Under Line: {game.Constant:.1f}")
                    st.write(
                        f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")
            ##elif selected_method == 'American':
                ##st.subheader('Coming soon - Decimal only')
elif selection == '📊 NHL Power Rankings':
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
        display_data = sorted_data[['powerranking', 'Team']].rename(columns={'Team': 'Team', 'powerranking': 'Power Ranking'})

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


elif selection == '🚫 NHL Injuries':
    excel_file = 'nhlgar.xlsx'
    sheet_name = 'Notable Injuries'
    # Load data from the specified sheet in the first Excel file
    game_data = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Sort the data based on the 'powerranking' column (assuming it's in column 'powerranking')
    sorted_data = game_data.sort_values(by='Rank')
    st.subheader("Important Injuries")
    st.write('All injuries are included in model and power rankings.')

         
       # Select columns to display
    columns_to_display = ['Rank', 'Player', 'Team', 'Injury']

    # Display the DataFrame without the index column
    st.table(sorted_data[columns_to_display].set_index('Rank'))
                    
elif selection == '🏀 NBA Model':

    st.title('NBA Model 🏀')
                                   
        # Use a relative path to the Excel file
    excel_file = 'nba.xlsx'

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
    # Define a list of available methods for calculating odds
    

    if st.button("Generate Todays's Odds", key="get_todays_odds"):
        
                    # Calculate the projected Money Line odds
                    today_games['Projected_Line'] = 0.2 * today_games['ml1'] + 0 * today_games['ml2'] + 0.8 * today_games['ml3']

                    today_games['Projected_Score'] = 0.6 * (today_games['homtot'] + today_games['vistot']) + 0.4 * (today_games['homtot1'] + today_games['vistot1'])   

                    today_games['Constant'] = np.round(today_games['Projected_Score'] / 0.5) * 0.5

                    # Set the standard deviation
                    std_deviation_overunder = 11.1
                    std_deviation_ml = 12

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
                    st.write("### Today's Games and Projected Odds:")
                    for i, game in enumerate(today_games.itertuples(), start=1):                   
                        st.subheader(f"{game.Visitor} *@* {game.Home}")
                        st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                        st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")

                        st.write(f"Projected Over Under Line: {game.Constant:.1f}")            
                        st.write(f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")

            
              
    if st.button("Generate Tomorrow's Odds", key="get_tomorrow_odds"):
            # Calculate and display the over/under odds, implied probabilities, and projected scores based on the selected method
            # Calculate the projected Money Line odds
            tomorrow_games['Projected_Line'] = 0.2 * tomorrow_games['ml1'] + 0 * tomorrow_games['ml2'] + 0.8 * tomorrow_games['ml3']

            tomorrow_games['Projected_Score'] = 0.6 * (tomorrow_games['homtot'] + tomorrow_games['vistot']) + 0.4 * (tomorrow_games['homtot1'] + tomorrow_games['vistot1'])

            # Round the constant to the nearest 0.5 using round_half_even
            tomorrow_games['Constant'] = np.round(tomorrow_games['Projected_Score'] / 0.5) * 0.5

                # Set the standard deviation
            std_deviation_overunder = 11.1
            std_deviation_ml = 12

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

                # Display the odds for tomorrow's games in a Streamlit table
            st.write("### Tomorrow's Games and Projected Odds:")

            for i, game in enumerate(tomorrow_games.itertuples(), start=1):
                    st.subheader(f"{game.Visitor} *@* {game.Home}")
                    st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                    st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")

                    st.write(f"Projected Over Under Line: {game.Constant:.1f}")
                    st.write(
                        f"**Over Under Odds:** Over: {game.Totals_Probability['Over']:.2f}, Under: {game.Totals_Probability['Under']:.2f}")

elif selection == '📊 NBA Power Rankings':
    excel_file = 'nba.xlsx'
    sheet_name = 'Powerrankings'
    sheet_name2 = "Topplayers"
      

    # Load data from the specified sheet in the first Excel file
    game_data = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Sort the data based on the 'powerranking' column (assuming it's in column 'powerranking')
    sorted_data = game_data.sort_values(by='Power')
    game_data2 = pd.read_excel(excel_file, sheet_name=sheet_name2)

    # Sort the data based on the 'topplayer' column (assuming it's in column 'topplayer')
    sorted_data2 = game_data2.sort_values(by='Rank')
    
    
        
    # Create a two-column layout
    col1, col2 = st.columns(2)  

  
        # Display content in the left column
    with col1:
        st.subheader("Team Rankings")

       # Rename the columns for display
        display_data = sorted_data[['Power', 'Team','Off Rank','Def Rank']].rename(columns={'Power': 'Ranking'})

        # Display the DataFrame without the index column
        st.dataframe(display_data, hide_index=True)

        st.subheader("Model's Top Players")

        display_data = sorted_data2[['Rank', 'topplayer', 'playteam', 'PTS', 'AST', 'REB','STL','BLK']].rename(columns={'topplayer': 'Player', 'playteam': 'Team'})

       
        # Display the DataFrame without the index column
        st.dataframe(display_data.head(15), hide_index=True)

             
    with col2:
        st.subheader("Models Top Rookies")
        display_data = sorted_data2[['Rank', 'bestrook', 'rookteam', 'p1', 'a1', 'r1','s1','b1']].rename(columns={'bestrook': 'Player', 'rookteam': 'Team', 'p1': 'PTS', 'a1': 'AST','r1': 'REB', 's1': 'STL', 'b1' : 'BLK'})
                           
        # Display the DataFrame without the index column
        st.dataframe(display_data.head(10), hide_index=True)

elif selection == '🚫 NBA Injuries':
    excel_file = 'nba.xlsx'
    sheet_name = 'notable injuries'
    # Load data from the specified sheet in the first Excel file
    game_data = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Sort the data based on the 'powerranking' column (assuming it's in column 'powerranking')
    sorted_data = game_data.sort_values(by='Rank')
    st.subheader("Important Injuries")
    st.write('All injuries are included in model and power rankings.')

         
       # Select columns to display
    columns_to_display = ['Rank', 'Player', 'Team', 'Injury']

    # Display the DataFrame without the index column
    st.table(sorted_data[columns_to_display].set_index('Rank'))

elif selection == '🔑 Betting Strategy':
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
     st.write('1. Unit size 2-5% of bank roll per bet')
     st.write('2. Don’t bet for the sake of betting - no inefficiencies no bet')
     st.write('3. Understand variance and how it impacts your returns')

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


    # Display the details of the last 10 bets
    st.subheader('Last 10 Bets')

    # Assuming you have a 'Bet Details' column in your DataFrame, adjust this accordingly
    last_10_bets_columns = ['Date1', 'Team', 'Performance', 'Odds Taken']  # Include 'Odds' in the list
    last_10_bets = df_outcomes.tail(10)[last_10_bets_columns]

    # Convert 'Date1' column to string and extract only the date part
    last_10_bets['Date'] = df_outcomes['Date1'].astype(str).str.split().str[0]

    # Rename columns
    last_10_bets.rename(columns={'Team': 'Bet', 'Performance': 'G/L', 'Odds Taken': 'Odds'}, inplace=True)

    # Convert 'G/L' column to numeric
    last_10_bets['G/L'] = pd.to_numeric(last_10_bets['G/L'], errors='coerce').round(1)

    # Drop the original 'Date1' column
    last_10_bets.drop(columns=['Date1'], inplace=True)

    # Reverse the order of rows
    last_10_bets = last_10_bets.iloc[::-1]

    # Highlight entire row based on conditions in any column
    def highlight_row(row):
        try:
            numeric_val = float(row['G/L'])
            color = 'background-color: green' if numeric_val > 0 else 'background-color: red' if numeric_val < 0 else ''
            return f'<tr style="{color}"><td>{row["Date"]}</td><td>{row["Bet"]}</td><td>{row["G/L"]}</td><td>{row["Odds"]}</td></tr>'
        except (ValueError, TypeError):
            return ''

    # Apply styling to the DataFrame
    styled_last_10_bets = ''.join(last_10_bets.apply(highlight_row, axis=1))

    # Display the HTML table with styling and headers using st.markdown
    table_html = f'<table><thead><tr><th>Date</th><th>Bet</th><th>G/L</th><th>Odds</th></tr></thead>{styled_last_10_bets}</table>'
    st.markdown(table_html, unsafe_allow_html=True)


