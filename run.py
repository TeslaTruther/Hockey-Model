import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from PIL import Image
import os
import scipy.stats as stats
import plotly.express as px

st.set_page_config(page_title="Hockey Locks", page_icon="🔒", layout="wide")




# Custom theme configurations
custom_theme = """
    [theme]
    primaryColor="#e62828"
    backgroundColor="#2f2f2f"
    secondaryBackgroundColor="#122687"
    textColor="#f9f8f8"
"""

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
selection = st.sidebar.radio('Hockey Locks 🔒', ['🏠 Home', '🏒 NHL Model', '🥅 NHL Power Rankings', '🏀 NBA Model', '💲Performance Tracking'])

if selection == '🏠 Home':
    # Main content
    st.title("Welcome to Hockey Locks :lock:")
    st.write("This app uses linear regression to generate odds using a combination of real time player and team data.")
    st.write("Use this tool to find inefficient odds and make positive EV bets.")   
     
    st.image(resized_pandas[0])      
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

        st.header("How the Model Works")
        st.write("The model generates odds from its projected probability of outcomes. Think of these odds as the minimum return you would require to make a positive EV bet.")
        st.subheader("Run The Model:")

        # Define a list of available methods for calculating odds
        ##calculation_methods = ['Decimal', 'American']

        # Add a selectbox to choose the calculation method
       ## selected_method = st.selectbox('Select Odds:', calculation_methods)

        # Apply custom CSS to make the select box smaller
        st.markdown(
            """
            <style>
            div[data-baseweb="select"] {
                max-width: 250px; /* Adjust the width as needed */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
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
elif selection == '🥅 NHL Power Rankings':
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
        display_data = sorted_data[['powerranking','Team']].rename(columns={'Team': 'Team', 'powerranking': 'Power Ranking'})
        st.dataframe(display_data, hide_index=True)

        st.subheader("Model's Top Goalies")
        display_data = sorted_data2[['Rank','topgoalie','golteam', 'gs','sv%','qs']].rename(columns={'topgoalie': 'Goalie', 'golteam': 'Team', 'gs' : 'Games Started','sv%':'SV%','qs':'Quality Starts'})
        # Display only the top 5 rows in a Streamlit dataframe
        st.dataframe(display_data.head(10), hide_index=True)

    # Display content in the right column
    with col2:
        st.subheader("Model's Top Players")
        # Rename the columns for display
         # Rename the columns for display and round 'mpg' to one decimal place
        display_data = sorted_data2[['Rank','topplayer', 'playteam','pos', 'gp', 'g', 'p', 'mpg']].rename(columns={'Rank': 'Rank','topplayer': 'Player', 'playteam': 'Team','pos': 'Pos', 'gp': 'GP', 'g':'Goals','p':'Points', 'mpg': 'MPG'})
        display_data['MPG'] = display_data['MPG'].round(1)  # Round 'mpg' to one decimal place
        # Display the sorted data in a Streamlit dataframe
        st.dataframe(display_data, hide_index=True,)

        st.subheader("Model's Top Rookies")
        display_data = sorted_data2[['Rank','bestrookies', 'rook team', 'pos1', 'gp1', 'g1','p1','mpg1']].rename(columns={'Rank': 'Rank','bestrookies': 'Rookie', 'rook team': 'Team', 'gp1': 'GP', 'pos1': 'Pos', 'mpg1': 'MPG','g1':'Goals','p1':'Points'})
        display_data['MPG'] = display_data['MPG'].round(1)  # Round 'mpg' to one decimal place
        # Display the sorted data in a Streamlit dataframe
        st.dataframe(display_data, hide_index=True,)




    
                    
elif selection == '🏀 NBA Model':

    st.title('Work In Progress - Odds are not sharp')
                                   
        # Use a relative path to the Excel file
    excel_file = 'nba.xlsx'

    # Load data from "Game Data" sheet
    game_data = pd.read_excel(excel_file, sheet_name="2023schedule")

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
    calculation_methods = ['Decimal', 'American']

    # Add a selectbox to choose the calculation method
    selected_method = st.selectbox('Select Odds:', calculation_methods)

    # Apply custom CSS to make the select box smaller
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] {
            max-width: 250px; /* Adjust the width as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Generate Todays's Odds", key="get_todays_odds"):
        
            if selected_method == 'Decimal':
                # Calculate the projected Money Line odds
                    today_games['Projected_Line'] =today_games['ml1'] 

                

                    # Set the standard deviation
                    std_deviation_ml = 14.2

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

                    
                    # Display the odds for today's games in a Streamlit table
                    st.write("### Today's Games and Projected Odds:")
                    for i, game in enumerate(today_games.itertuples(), start=1):                   
                        st.subheader(f"{game.Visitor} *@* {game.Home}")
                        st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                        st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")
            
            elif selected_method == 'American':
                st.subheader('Coming soon - Decimal only')  
# Button to get today's odds
    if st.button("Generate Tomorrow's Odds", key="get_tomorrow_odds"):
        # Calculate and display the over/under odds, implied probabilities, and projected scores based on the selected method
            if selected_method == 'Decimal':
                # Calculate and display the over/under odds, implied probabilities, and projected scores
                

                # Calculate the projected Money Line odds
                tomorrow_games['Projected_Line'] =tomorrow_games['ml1'] 

            

                # Set the standard deviation
                std_deviation_ml = 14.2

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

                
                # Display the odds for today's games in a Streamlit table
                st.write("### Today's Games and Projected Odds:")
                for i, game in enumerate(tomorrow_games.itertuples(), start=1):                   
                    st.subheader(f"{game.Visitor} *@* {game.Home}")
                    st.write(f"{game.Home} | **Projected Odds:** {game.ML_Home_Decimal_Odds:.3f}")
                    st.write(f"{game.Visitor} | **Projected Odds:** {game.ML_Away_Decimal_Odds:.3f}")
                

            elif selected_method == 'American':
                st.subheader('Coming Soon - Decimal Only')  
    

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

    
    file_path = 'EV Performance.xlsx'  

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


