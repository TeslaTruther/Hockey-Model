from datetime import datetime
import requests
import sqlite3

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

def create_table():
    # Connect to the database (creates 'nhldata.db' if it doesn't exist)
    conn = sqlite3.connect('nhldata.db')
    cursor = conn.cursor()

    # Drop the odds_data table if it exists
    cursor.execute('''DROP TABLE IF EXISTS odds_data''')

    # Create the odds_data table with the specified schema
    cursor.execute('''
        CREATE TABLE odds_data (
            id INTEGER PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            commence_time TEXT,
            bookmaker TEXT,
            home_win_odds REAL,
            away_win_odds REAL,
            total_number_goals REAL,
            total_over_odds REAL,
            total_under_odds REAL
        )
    ''')

    # Commit and close the connection
    conn.commit()
    conn.close()

def create_nba_table():
    # Connect to the database (creates 'nbadata.db' if it doesn't exist)
    conn = sqlite3.connect('nbadata.db')
    cursor = conn.cursor()

    # Drop the odds_data table if it exists
    cursor.execute('''DROP TABLE IF EXISTS odds_data''')

    # Create the odds_data table with the specified schema
    cursor.execute('''
        CREATE TABLE odds_data (
            id INTEGER PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            commence_time TEXT,
            bookmaker TEXT,
            home_win_odds REAL,
            away_win_odds REAL,
            total_number_goals REAL,
            total_over_odds REAL,
            total_under_odds REAL
        )
    ''')

    # Commit and close the connection
    conn.commit()
    conn.close()
create_nba_table()

def fetch_nba_data():
    # Connect to the database (creates 'nbadata.db' if it doesn't exist)
    conn = sqlite3.connect('nbadata.db')
    cursor = conn.cursor()

    # Your Odds API key
    API_KEY = "0b8b0f798933d0c1e0ba7a7228ec21fa"

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

    params = {
        "apiKey": API_KEY,
        "regions": "us",  # Focus on US bookmakers
        "markets": "h2h,totals",  # Get moneyline and totals odds
        "oddsFormat": "decimal"  # Request odds in decimal format
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Check if any odds are found
        if data:
            print("NBA Odds:")
            for game in data:
                home_team = game['home_team']
                away_team = game['away_team']
                commence_time = datetime.strptime(game['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                commence_time_str = commence_time.strftime("%Y-%m-%d %H:%M:%S")  # Format commence time string

                # Find bookmakers and their odds
                bookmakers = game['bookmakers']
                for bookmaker in bookmakers:
                    if bookmaker['title'] == 'FanDuel':
                        markets = bookmaker['markets']
                        home_win_odds = None
                        away_win_odds = None
                        total_over_odds = None
                        total_under_odds = None
                        total_number_goals = None
                        for market in markets:
                            if market['key'] == 'h2h':
                                home_win_odds = market['outcomes'][0]['price']
                                away_win_odds = market['outcomes'][1]['price']
                            elif market['key'] == 'totals':
                                outcomes = market['outcomes']
                                for outcome in outcomes:
                                    if outcome['name'] == 'Over':
                                        total_over_odds = outcome['price']
                                        total_number_goals = outcome['point']
                                    elif outcome['name'] == 'Under':
                                        total_under_odds = outcome['price']

                        # Insert data into the database
                        cursor.execute('''
                            INSERT INTO odds_data (home_team, away_team, commence_time, bookmaker, home_win_odds, away_win_odds, total_number_goals, total_over_odds, total_under_odds)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (home_team, away_team, commence_time_str, bookmaker['title'], home_win_odds, away_win_odds, total_number_goals, total_over_odds, total_under_odds))

            # Save (commit) the changes
            conn.commit()
            print_database(conn)
            
        else:
            print("No NBA odds found currently.")
    else:
        print("Request failed. Status code:", response.status_code)

    # Close the database connection
    conn.close()

fetch_nba_data()

def fetch_and_insert_data():
    # Connect to the database (creates 'nhldata.db' if it doesn't exist)
    conn = sqlite3.connect('nhldata.db')
    cursor = conn.cursor()

    # Your Odds API key
    API_KEY = "0b8b0f798933d0c1e0ba7a7228ec21fa"

    # Endpoint for NHL odds
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"

    # Parameters for the API request
    params = {
        "apiKey": API_KEY,
        "regions": "us",  # Focus on US bookmakers
        "markets": "h2h,totals",  # Get moneyline and totals odds
        "oddsFormat": "decimal"  # Request odds in decimal format
    }

    # Send the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Check if any odds are found
        if data:
            print("NHL Odds:")
            for game in data:
                home_team = game['home_team']
                away_team = game['away_team']
                commence_time = datetime.strptime(game['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                commence_time_str = commence_time.strftime("%Y-%m-%d %H:%M:%S")  # Format commence time string

                # Find bookmakers and their odds
                bookmakers = game['bookmakers']
                for bookmaker in bookmakers:
                    if bookmaker['title'] == 'FanDuel':
                        markets = bookmaker['markets']
                        home_win_odds = None
                        away_win_odds = None
                        total_over_odds = None
                        total_under_odds = None
                        total_number_goals = None
                        for market in markets:
                            if market['key'] == 'h2h':
                                home_win_odds = market['outcomes'][0]['price']
                                away_win_odds = market['outcomes'][1]['price']
                            elif market['key'] == 'totals':
                                outcomes = market['outcomes']
                                for outcome in outcomes:
                                    if outcome['name'] == 'Over':
                                        total_over_odds = outcome['price']
                                        total_number_goals = outcome['point']
                                    elif outcome['name'] == 'Under':
                                        total_under_odds = outcome['price']

                        # Insert data into the database
                        cursor.execute('''
                            INSERT INTO odds_data (home_team, away_team, commence_time, bookmaker, home_win_odds, away_win_odds, total_number_goals, total_over_odds, total_under_odds)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (home_team, away_team, commence_time_str, bookmaker['title'], home_win_odds, away_win_odds, total_number_goals, total_over_odds, total_under_odds))

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
