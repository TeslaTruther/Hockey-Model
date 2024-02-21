import sqlite3
from datetime import datetime
import requests
import time

def print_database(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM odds_data")
    rows = cursor.fetchall()

    if rows:
      print("Database Contents:")
      for row in rows:
          print(row)
    else:
      print("Database is currently empty.")
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
                commence_time_str = commence_time.strftime("%Y-%m-%d %H:%M:%S")  # Format commence time string

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
                                ''', (home_team, away_team, commence_time_str, bookmaker['title'], home_win_odds, away_win_odds))

            # Save (commit) the changes
            conn.commit()

            print_database(conn)
        else:
            print("No NHL odds found currently.")
    else:
        print("Request failed. Status code:", response.status_code)

    # Close the database connection
    conn.close()

fetch_and_insert_data()