import sqlite3
from datetime import datetime
import requests

def print_database():
    # Connect to the database
    conn = sqlite3.connect('mydata.db')
    cursor = conn.cursor()

    # Execute a SELECT query to fetch all data from the 'odds_data' table
    cursor.execute("SELECT * FROM odds_data ORDER BY commence_time")
    rows = cursor.fetchall()

    # Print the database contents
    if rows:
        print("Database Contents:")
        for row in rows:
            print(row)
    else:
        print("Database is currently empty.")

    # Close the database connection
    conn.close()

# Function to fetch and insert data into the database
def fetch_and_insert_data():
        # Connect to the database (creates 'mydata.db' if it doesn't exist)
        conn = sqlite3.connect('mydata.db')
        cursor = conn.cursor()

        # Your Odds API key
        API_KEY = "0b8b0f798933d0c1e0ba7a7228ec21fa"

        # Endpoint for NHL odds
        url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"

        # Parameters for the API request
        params = {
            "apiKey": API_KEY,
            "regions": "us",  # Focus on US bookmakers
            "markets": "h2h",  # Get moneyline odds
            "oddsFormat": "decimal"  # Request odds in decimal format
        }

        # Send the API request
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Check if any odds are found
            if data:
                print("NHL Moneyline Odds:")
                for game in data:
                    home_team = game['home_team']
                    away_team = game['away_team']
                    commence_time = datetime.strptime(game['commence_time'], "%Y-%m-%dT%H:%M:%SZ")

                    # Find bookmakers and their odds
                    bookmakers = game['bookmakers']
                    for bookmaker in bookmakers:
                        # Focus on moneyline markets (h2h) and FanDuel bookmaker
                        if bookmaker['title'] == 'FanDuel':
                            markets = bookmaker['markets']
                            for market in markets:
                                if market['key'] == 'h2h':
                                    home_win_odds = market['outcomes'][0]['price']
                                    away_win_odds = market['outcomes'][1]['price']
                                    # Insert data into the database
                                    cursor.execute('''
                                        INSERT INTO odds_data (home_team, away_team, commence_time, bookmaker, home_win_odds, away_win_odds)
                                        VALUES (?, ?, ?, ?, ?, ?)
                                    ''', (home_team, away_team, commence_time, bookmaker['title'], home_win_odds, away_win_odds))
                                    
                                    # Check if the odds correspond to the current projected game
                                    # If you want to write it to a streamlit app, you need to import and use st.write()
                                    # You also need to have the 'game' object available in your Streamlit script
                                    # if home_team == game.Home and away_team == game.Visitor:
                                    #     st.write(f"{game.Home}: {home_win_odds}, {game.Visitor}: {away_win_odds}")
                                    # else:
                                    #     print("No NHL odds found currently.")

                # Save (commit) the changes
                conn.commit()
                

            else:
                print("No NHL odds found.")

        else:
            print("Request failed. Status code:", response.status_code)

        # Close the database connection
        conn.close()

def fetch_and_insert_totals_data():
    # Connect to the database (creates 'mydata.db' if it doesn't exist)
    conn = sqlite3.connect('mydata.db')
    cursor = conn.cursor()

    # Your Odds API key
    API_KEY = "0b8b0f798933d0c1e0ba7a7228ec21fa"

    # Endpoint for NHL odds (totals lines)
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"

    # Parameters for the API request (focus on totals lines)
    params = {
        "apiKey": API_KEY,
        "regions": "us",  # Focus on US bookmakers
        "markets": "totals",  # Get totals lines
        "oddsFormat": "decimal"  # Request odds in decimal format
    }

    # Send the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Check if any totals lines are found
        if data:
            print("NHL Totals Lines:")
            for game in data:
                event_name = game['event_name']
                commence_time = datetime.strptime(game['commence'], "%Y-%m-%dT%H:%M:%S.%fZ")
                status = game['status']
                bookmaker = game['bookmaker']
                last_update = datetime.strptime(game['last_update'], "%Y-%m-%dT%H:%M:%S.%fZ")
                point_1 = game['point_1']
                point_2 = game['point_2']
                odd_1 = game['odd_1']
                odd_2 = game['odd_2']

                # Insert data into the database
                cursor.execute('''
                    INSERT INTO odds_data (event_name, commence_time, status, bookmaker, last_update, point_1, point_2, odd_1, odd_2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (event_name, commence_time, status, bookmaker, last_update, point_1, point_2, odd_1, odd_2))
                print("Inserted data:", event_name)  # Debugging print statement

            # Save (commit) the changes
            conn.commit()

            # Print the database contents
            print_database()

        else:
            print("No NHL totals lines found.")

    else:
        print("Request failed. Status code:", response.status_code)

    # Close the database connection
    conn.close()
