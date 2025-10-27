import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm.auto import tqdm
import os
import bar_chart_race as bcr
import imageio_ffmpeg as ffmpeg
import matplotlib as mpl

# Set the path for ffmpeg so matplotlib can export animations as videos
mpl.rcParams['animation.ffmpeg_path'] = ffmpeg.get_ffmpeg_exe()

def load_games():
    """
    Load and clean games data from a CSV file.

    Combines city and team names for home and away teams, 
    converts game dates to datetime, and selects relevant columns.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame with game information.
    """
     
    # Read the CSV file containing game data into a DataFrame
    df_games = pd.read_csv('inputs\\Games.csv')

    # Create a 'home_team' column by combining city and team name for the home team
    df_games['home_team'] = df_games['hometeamCity'].str.strip() + ' ' + df_games['hometeamName'].str.strip()

    # Create an 'away_team' column by combining city and team name for the away team
    df_games['away_team'] = df_games['awayteamCity'].str.strip() + ' ' + df_games['awayteamName'].str.strip()

    # Convert 'gameDate' column to datetime format
    df_games['gameDate'] = pd.to_datetime(df_games['gameDate'])

    # Extract only the date part from 'gameDate' (without time) into a new column 'game_date'
    df_games['game_date'] = df_games['gameDate'].dt.date

    # Keep only the relevant columns for analysis
    df_games=df_games[['game_date', 'home_team', 'away_team', 'homeScore', 'awayScore', 'gameType']]
    return df_games

def load_players():
    """
    Load and clean player statistics data from a CSV file.

    Combines first and last names, constructs team columns, 
    converts minutes into a usable format, and extracts unique dates and players.

    Returns:
        df_players (pandas.DataFrame): Cleaned player statistics.
        sorted_dates (list): Sorted list of unique game dates.
        players (list): List of unique player names.
    """

    # Read the player statistics CSV into a DataFrame
    df_players = pd.read_csv('inputs\\PlayerStatistics.csv')

    # Convert 'gameDate' column to datetime format
    df_players['gameDate'] = pd.to_datetime(df_players['gameDate'])

    # Extract only the date (without time) into a new column
    df_players['game_date'] = df_players['gameDate'].dt.date

    # Get a sorted list of all unique game dates
    sorted_dates = sorted(df_players['game_date'].unique())

    # Create a 'fullName' column by combining first and last name
    df_players['fullName'] = df_players['firstName'].str.strip() + ' ' + df_players['lastName'].str.strip()

    # Create 'player_team' and 'opponent_team' columns combining city and team name
    df_players['player_team'] = df_players['playerteamCity'].str.strip() + ' ' + df_players['playerteamName'].str.strip()
    df_players['opponent_team'] = df_players['opponentteamCity'].str.strip() + ' ' + df_players['opponentteamName'].str.strip()

    # Replace infinite values in 'numMinutes' with NaN, then fill NaN with 0
    df_players['numMinutes'] = df_players['numMinutes'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Split 'numMinutes' into whole minutes and seconds
    minutes = df_players['numMinutes'].astype(int)
    seconds = ((df_players['numMinutes'] - minutes) * 100).round()

    # Combine minutes and seconds into total minutes as float
    df_players['minutes']= minutes+seconds/60

    # Get a list of all unique player full names
    players = df_players['fullName'].unique()

    # Keep only relevant columns for analysis
    df_players = df_players[['game_date','fullName','player_team', 'opponent_team', 'minutes', 'gameId', 'gameType']]

    # Return cleaned DataFrame, sorted dates, and list of players
    return df_players, sorted_dates, players

def load_files():
    """
    Load previously saved ELO data to resume processing from the last checkpoint.

    This function checks if there are any saved files containing ELO history.
    If found, it automatically loads:
        - Full ELO history per player (elo_history)
        - Last ELO snapshot per player (last_ELO)
        - Highest ELO ever achieved by each player (highest_ELO)
        - Team ELO values associated with each player per game (team_player_ELO)
        - Opponent team ELO values associated with each player per game (opponent_team_player_ELO)

    Returns:
        tuple:
            elo_history (dict): {player_name: [ELO values over time]}
            last_ELO (dict): {player_name: last recorded ELO value}
            highest_ELO (dict): {player_name: highest ELO recorded}
            team_player_ELO (dict): {player_name: [ELO of player's team per game]}
            opponent_team_player_ELO (dict): {player_name: [ELO of opponent team per game]}
            OR
            None (if no saved files exist)
    """
    # --- LOAD CHECKPOINT IF EXISTS ---
    # Check if there are any previously saved ELO files
    save_dir = 'data'
    saved_files = [f for f in os.listdir(save_dir) if f.startswith("elo_history_until_")]
    if saved_files:
        # Find last saved date
        # 1. Sort the list and take the last one (latest date) 
        latest_file = sorted(saved_files)[-1]
        # 2. Split by "_until_" and then by ".csv" to get only the date string
        last_saved_date = latest_file.split("_until_")[1].split(".csv")[0]

        print(f"🔁 Resuming from {last_saved_date}...")

        # Load full ELO history from the latest saved file
        elo_history_df = pd.read_csv(os.path.join(save_dir, latest_file))
        # Convert the DataFrame into a dictionary with player names as keys and lists of ELO values as values (the types used in the code)
        elo_history = {player: elo_history_df[player].dropna().tolist() for player in elo_history_df.columns}
  

        # Load the last ELO values per player from CSV
        last_elo_df = pd.read_csv(os.path.join(save_dir, f"last_elo_until_{last_saved_date}.csv"))
        # Convert DataFrame to dictionary to match the types used in the code: player -> last ELO
        last_ELO = dict(zip(last_elo_df['player'], last_elo_df['last_ELO']))

        # Load the highest ELO achieved by each player
        highest_elo_df = pd.read_csv(os.path.join(save_dir, f"highest_elo_until_{last_saved_date}.csv"))
        highest_ELO = dict(zip(highest_elo_df['player'], highest_elo_df['highest_ELO']))

        # Load team ELO per player
        team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"team_elo_until_{last_saved_date}.csv"))
        team_player_ELO = {player: team_player_ELO_df[player].tolist() for player in team_player_ELO_df.columns}

        # Load opponent team ELO per player
        opponent_team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"opponent_team_until_{last_saved_date}.csv"))
        opponent_team_player_ELO = {player: opponent_team_player_ELO_df[player].tolist() for player in opponent_team_player_ELO_df.columns}

        # Return all recovered structures
        return elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO

def save(last_date, elo_history, last_ELO, highest_ELO, team_ELO, opponent_team_ELO):
    """
    Save ELO data to CSV files in the 'data' folder.

    Saves full ELO history, last ELO per player, highest ELO per player,
    team ELO, and opponent team ELO up to a given date.

    Parameters:
        last_date (str or date): The last date of the data being saved.
        elo_history (dict): Dictionary containing full ELO history for players.
        last_ELO (dict): Dictionary with the latest ELO per player.
        highest_ELO (dict): Dictionary with the highest ELO per player.
        team_ELO (dict): Dictionary of team ELO per player.
        opponent_team_ELO (dict): Dictionary of opponent team ELO per player.
    """

    save_dir = 'data'

    # Convert the full ELO history dictionary into a DataFrame and save it as a CSV file
    # `orient='index'` means keys of the dictionary become row labels
    # `transpose()` flips rows and columns so players become columns and dates become rows
    elo_history_df = pd.DataFrame.from_dict(elo_history, orient='index').transpose()
    elo_history_df.to_csv(os.path.join(save_dir, f"elo_history_until_{last_date}.csv"), index=False)

    # Convert the dictionary of last ELO values per player into a DataFrame and save it as a CSV file
    # `list(last_ELO.items())` turns the dict into a list of (player, ELO) tuples
    last_elo_df = pd.DataFrame(list(last_ELO.items()), columns=['player', 'last_ELO'])
    last_elo_df.to_csv(os.path.join(save_dir, f"last_elo_until_{last_date}.csv"), index=False)

    # Save highest ELO per player
    highest_elo_df = pd.DataFrame(list(highest_ELO.items()), columns=['player', 'highest_ELO'])
    highest_elo_df.to_csv(os.path.join(save_dir, f"highest_elo_until_{last_date}.csv"), index=False)

    # Save team ELO per player
    team_elo_df = pd.DataFrame.from_dict(team_ELO, orient='index').transpose()
    team_elo_df.to_csv(os.path.join(save_dir, f"team_elo_until_{last_date}.csv"), index=False)

    # Save opponent team ELO per player
    opponent_team_ELO_df = pd.DataFrame.from_dict(opponent_team_ELO, orient='index').transpose()
    opponent_team_ELO_df.to_csv(os.path.join(save_dir, f"opponent_team_until_{last_date}.csv"), index=False)
        
def calculate_team_ELO(players_rows, last_ELO):
    """
    Calculate the weighted average ELO for a team in a game based on player minutes.

    Parameters:
        players_rows (pandas.DataFrame): DataFrame containing player stats for a game, including 'minutes' and 'fullName'.
        last_ELO (dict): Dictionary mapping player names to their latest ELO rating.

    Returns:
        tuple: (team_avg_ELO, total_minutes, minutes_per_player)
            - team_avg_ELO: Weighted average ELO of the team for this game.
            - total_minutes: Sum of minutes played by all players.
            - minutes_per_player: If default minutes are used due to missing data this is that value, otherwise None.
    """
    # Sum up the minutes played by all players in the given game
    total_minutes = players_rows['minutes'].sum()

     # Count how many players are present in the rows
    num_players = len(players_rows)

    # Check for invalid input: there must be at least one player
    # If not, raise an AssertionError showing the problematic DataFrame
    assert num_players > 0, f"{players_rows}"

    # If the total minutes are zero or there are fewer than 5 players, use default values
    if total_minutes <= 0 or num_players < 5:
        # Default total minutes: 48 minutes per player * 5 players (full game)
        total_minutes = 48 * 5

        # Distribute minutes evenly among all available players
        minutes_per_player = total_minutes / num_players

        # Compute weighted sum of ELOs using the default minutes
        # iterrows() loops over each row, multiply the player's ELO by the assigned minutes
        weighted_sum = sum(last_ELO[row['fullName']] * minutes_per_player for _, row in players_rows.iterrows())

        # Return weighted average ELO, total minutes, and minutes per player
        return weighted_sum / total_minutes, total_minutes, minutes_per_player
    else:
        # Compute weighted sum of ELOs based on actual minutes played
        weighted_sum = sum(last_ELO[row['fullName']] * row['minutes'] for _, row in players_rows.iterrows())

        # Return weighted average ELO, total minutes, and None for minutes_per_player since the information is available
        return weighted_sum / total_minutes, total_minutes, None

def elo_computation(df_games, df_players, sorted_dates, players):
    """
    Compute player ELO ratings over time based on game results.

    Parameters:
        df_games (pandas.DataFrame): DataFrame with game info (home/away teams, scores, game type).
        df_players (pandas.DataFrame): DataFrame with player stats including minutes played.
        sorted_dates (list): Sorted list of game dates.
        players (list): List of all player full names.

    Returns:
        tuple: (elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO)
            - elo_history: Dict with full ELO history per player.
            - last_ELO: Dict with latest ELO per player.
            - highest_ELO: Dict with highest ELO reached per player.
            - team_player_ELO: Dict with ELO of the player's team per game.
            - opponent_team_player_ELO: Dict with ELO of the opponent team per game.
    """

    # Create 'data' directory if it doesn't exist
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)

    # --- LOAD CHECKPOINT IF EXISTS ---
    # Check if there are any previously saved ELO files
    saved_files = [f for f in os.listdir(save_dir) if f.startswith("elo_history_until_")]
    if saved_files:
        # Find last saved date
        # 1. Sort the list and take the last one (latest date) 
        latest_file = sorted(saved_files)[-1]
        # 2. Split by "_until_" and then by ".csv" to get only the date string
        last_saved_date = latest_file.split("_until_")[1].split(".csv")[0]

        print(f"🔁 Resuming from {last_saved_date}...")

        # Load full ELO history from the latest saved file
        elo_history_df = pd.read_csv(os.path.join(save_dir, latest_file))

        # Convert the DataFrame into a dictionary with player names as keys and lists of ELO values as values (the types used in the code)
        elo_history = {player: elo_history_df[player].dropna().tolist() for player in elo_history_df.columns}
        # Store the number of rows in the ELO history (used to track length)
        length = len(elo_history_df)

        # Load the last ELO values per player from CSV
        last_elo_df = pd.read_csv(os.path.join(save_dir, f"last_elo_until_{last_saved_date}.csv"))
        # Convert DataFrame to dictionary to match the types used in the code: player -> last ELO
        last_ELO = dict(zip(last_elo_df['player'], last_elo_df['last_ELO']))

        # Load the highest ELO achieved by each player
        highest_elo_df = pd.read_csv(os.path.join(save_dir, f"highest_elo_until_{last_saved_date}.csv"))
        # Convert DataFrame to dictionary to match the types used in the code: player -> highest ELO
        highest_ELO = dict(zip(highest_elo_df['player'], highest_elo_df['highest_ELO']))

        # Load team ELO per player
        team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"team_elo_until_{last_saved_date}.csv"))
        # Convert to dictionary: player -> list of team ELOs over time
        team_player_ELO = {player: team_player_ELO_df[player].tolist() for player in team_player_ELO_df.columns}

        # Load opponent team ELO per player
        opponent_team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"opponent_team_until_{last_saved_date}.csv"))
        # Convert to dictionary: player -> list of opponent team ELOs over time
        opponent_team_player_ELO = {player: opponent_team_player_ELO_df[player].tolist() for player in opponent_team_player_ELO_df.columns}

        # Load the list of players using the last ELO dictionary
        players = list(last_ELO.keys())

        # Filter dates to only include games after the last saved date
        sorted_dates_filtered = [d for d in sorted_dates if d > datetime.datetime.strptime(last_saved_date, "%Y-%m-%d").date()]

        # Filter the players DataFrame to only include entries after the last saved date
        df_players_filtered = df_players[df_players['game_date']> datetime.datetime.strptime(last_saved_date, "%Y-%m-%d").date()].copy()
        
        # Filter the games DataFrame to only include entries after the last saved date
        df_games_filtered = df_games[df_games['game_date']> datetime.datetime.strptime(last_saved_date, "%Y-%m-%d").date()].copy()

        # Set the current game date to the last saved date (resume point)
        game_date = last_saved_date
    else:
        print("🚀 Initializing from scratch...")

        # Initialize ELO history dictionary: each player starts with 1000 as the first ELO value
        elo_history = {player: [1000] for player in players} 

        # Initialize last ELO dictionary: stores the most recent ELO for each player
        last_ELO = {player: 1000 for player in players} 

        # Initialize highest ELO dictionary: tracks the highest ELO each player has achieved
        highest_ELO = {player: 1000 for player in players}

        # Initialize opponent team ELO history per player (None for each date in which the players doesn't play a game)
        team_player_ELO = {player: [None] for player in players} 

        # Initialize team ELO history per player (None for each date in which the players doesn't play a game)
        opponent_team_player_ELO = {player: [None] for player in players} # Team ELO of the player's opponent team

        # Initialize length of ELO history
        length = 1 

        # Create copies of the DataFrames
        df_players_filtered=df_players.copy()
        df_games_filtered=df_games.copy()

        # Use the full list of sorted game dates
        sorted_dates_filtered = sorted_dates

    # Set 'game_date' as the index for easier selection of games by date
    df_games_filtered.set_index('game_date', inplace=True)
    df_players_filtered.set_index('game_date', inplace=True)


    # Main loop through all game dates
    for game_date in tqdm(sorted_dates_filtered):
        # Select all games and players for the current date
        daily_games = df_games_filtered.loc[[game_date]]
        players_today = df_players_filtered.loc[[game_date]]

        # Loop through each game of the day
        for game in daily_games.itertuples(index=False):
            # Extract home and away teams
            home_team = game.home_team
            away_team = game.away_team

            # Group players by team for this game
            team_players = {
                home_team: players_today[players_today['player_team'] == home_team],
                away_team: players_today[players_today['player_team'] == away_team]
            }

            # Dictionaries to store team ELO, total minutes, and default minutes (if data missing). Indexed by team name.
            team_ELO = {}
            team_played_time = {}
            default_minutes = {}

            # Calculate average ELO and total minutes for each team
            for team in [home_team, away_team]:
                players_rows = team_players[team]
                # `calculate_team_ELO` returns: (weighted_avg_ELO, total_minutes, default_minutes)
                team_ELO[team], team_played_time[team], default_minutes[team] = calculate_team_ELO(players_rows, last_ELO)

            # Calculate expected win probabilities using the ELO difference
            E_home = 1 / (1 + 10 ** ((team_ELO[away_team] - team_ELO[home_team]) / 400))
            E_away = 1 - E_home
            team_E = {home_team: E_home, away_team: E_away}

            # Determine K factor (larger for playoffs, smaller for regular season games)
            K = 32 if game.gameType == 'Playoffs' else 16

            # Update each player's ELO based on the game result
            for team in [home_team, away_team]:
                # Determine if the team won the game
                won = (team == home_team and game.homeScore > game.awayScore) or \
                    (team == away_team and game.awayScore > game.homeScore)
                S = 1 if won else 0  # Actual score: 1 if win, 0 if loss
                E = team_E[team] # Expected score from ELO

                players_rows = team_players[team]

                # Loop through each player in the team
                for row in players_rows.itertuples(index=False):
                    player = row.fullName
                    current_ELO = last_ELO[player]

                    # Use default minutes if available
                    minutes_played = row.minutes if default_minutes[team] is None else default_minutes[team]

                    # Calculate ELO change based on player's minutes and game outcome (per 36 min)
                    delta =  (minutes_played / 36) * K * (S - E)
                    new_ELO = current_ELO + delta

                    # Update player ELO history
                    elo_history[player].append(new_ELO) # Full ELO history
                    last_ELO[player] = new_ELO # Latest ELO
                    highest_ELO[player] = max(highest_ELO[player], new_ELO) # Track highest ELO

                    # Save team and opponent team ELO for this game
                    team_player_ELO[player].append(team_ELO[home_team] if team == home_team else team_ELO[away_team])
                    opponent_team_player_ELO[player].append(team_ELO[away_team] if team == home_team else team_ELO[home_team])
                    

        # Remove the current game date from the filtered DataFrames to free memory and speed up future computations
        # `errors='ignore'` prevents errors if the date is missing (already dropped)
        df_games_filtered.drop(index=game_date, inplace=True, errors='ignore')
        df_players_filtered.drop(index=game_date, inplace=True, errors='ignore')

        # Increment the length of ELO history, tracking the number of processed time points
        length += 1
        # Ensure that all players have the same number of ELO entries (some players may not have played on this date)
        for player in elo_history:
            if len(elo_history[player]) < length:
                # Carry forward the last known ELO for players who didn't play
                elo_history[player].append(last_ELO[player])
                # Append None for team and opponent team ELO since the player didn't play
                team_player_ELO[player].append(None)
                opponent_team_player_ELO[player].append(None)


        # Show progress every 100 processed dates
        if length % 100 == 0:
            print(f"\n📅 Date: {game_date}")

            # Sort players by highest ELO achieved and by current ELO, take top 10
            top_highest = sorted(highest_ELO.items(), key=lambda x: x[1], reverse=True)[:10]
            top_current = sorted(last_ELO.items(), key=lambda x: x[1], reverse=True)[:10]

            # Print header with fixed-width columns for readability
            name_width = 20  # Adjust column width as needed
            print(
                f"{'🏆 Highest Player ELO 🏆':<{name_width}}        "
                f"{'🔥 Current Player ELO 🔥':<{name_width}}"
            )

            # Print rows showing top 10 highest and current ELOs side by side
            for (high_name, high_elo), (curr_name, curr_elo) in zip(top_highest, top_current):
                print(
                    f"{high_name:<{name_width}} {high_elo:>7.2f}    "
                    f"{curr_name:<{name_width}} {curr_elo:>7.2f}"
                )

            # Save intermediate progress every 1000 dates to avoid losing data
            if length % 1000 == 0:
                print(f"\n💾 Saving progress at date {game_date}...")
                save(game_date, elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO)

    # Final save at the end of computation
    print(f"\n💾 Saving progress at date {game_date}...")
    save(game_date, elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO)

    # Return all ELO structures
    return elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO

def get_yearly_quarterly_monthly_graphs(mode, df, title):
    """
    Generate bar chart race animations for ELO ratings aggregated yearly, quarterly, and monthly.

    Parameters:
        mode (str): Identifier for the type of data (used in filename).
        df (pandas.DataFrame): DataFrame with player ELOs, indexed by date.
        title (str): Title prefix for the charts.

    Produces:
        Three MP4 videos saved in the 'outputs' folder: yearly, quarterly, and monthly ELO bar chart races.
    """

    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index) 

    # ------------------ YEARLY ------------------
    # Resample the data by year and compute the mean ELO per year
    df_active_players_year_avg = df.resample('Y').mean()
    # Remove rows where all values are NaN 
    df_active_players_year_avg.dropna(how='all', inplace=True)
    # Create bar chart race for yearly ELOs
    bcr.bar_chart_race(
        df=df_active_players_year_avg.fillna(0), # Fill missing values with 0
        filename='outputs\\yearly_' + mode + '_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.mp4',      
        orientation='h',                         # Horizontal bars
        sort='desc',                             # Sort descending
        n_bars=15,                               # Show top 15 players
        fixed_order=False,                       # Allow bars to move dynamically
        fixed_max=True,                          # Keep max consistent across periods
        steps_per_period=20,                     # Smooth animation
        period_length=1000,                      # Duration per period in milliseconds
        title='Yearly ELO Rating - ' + title,    # Title
        interpolate_period=False,                # No interpolation
        bar_size=.8,
        cmap='dark12',                           # Color map
        dpi=144,
        period_label={'x': 0.99, 'y': 0.1, 'ha': 'right', 'va': 'center'},
    )

    # ------------------ QUARTERLY ------------------
    # Resample by quarter and compute mean ELO per quarter
    df_active_players_quarter_avg = df.resample('Q').mean() 
    df_active_players_quarter_avg.dropna(how='all', inplace=True)
    # Create bar chart race for quarterly ELOs: same configuartion 
    bcr.bar_chart_race(
        df=df_active_players_quarter_avg.fillna(0),
        filename='outputs\\quarterly_' + mode + '_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.mp4',      
        orientation='h',               
        sort='desc',
        n_bars=15,
        fixed_order=False,
        fixed_max=True,
        steps_per_period=20,
        period_length=1000,
        title='Quarterly ELO Rating - ' + title,
        interpolate_period=False,
        bar_size=.8,
        cmap='dark12',
        dpi=144,
        period_label={'x': 0.99, 'y': 0.1, 'ha': 'right', 'va': 'center'},
    )

    # ------------------ MONTHLY ------------------
    # Resample by month and compute mean ELO per month
    df_active_players_month_avg = df.resample('M').mean() 
    df_active_players_month_avg.dropna(how='all', inplace=True)
    # Create bar chart race for monthly ELOs: same configuration
    bcr.bar_chart_race(
        df=df_active_players_month_avg.fillna(0),
        filename='outputs\\monthly_' + mode + '_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.mp4',      
        orientation='h',               
        sort='desc',
        n_bars=15,
        fixed_order=False,
        fixed_max=True,
        steps_per_period=20,
        period_length=1000,
        title='Monthly ELO Rating - ' + title,
        interpolate_period=False,
        bar_size=.8,
        cmap='dark12',
        dpi=144,
        period_label={'x': 0.99, 'y': 0.1, 'ha': 'right', 'va': 'center'},
    )

def elo_historic_rating_players(elo_history_df, highest_elo_df):
    """
    Generate bar chart race animations for historic player ELO ratings. Generate yearly,
    quarterly, and monthly animations for both:
            - Historic ELO
            - Maximum ELO reached over time

    Parameters:
        elo_history_df (pd.DataFrame): ELO history with dates as index and players as columns
        highest_elo_df (pd.DataFrame): DataFrame with columns ['fullName','highest_ELO']

    Returns:
        None (outputs animations to 'outputs' folder)
    """

    # Unpivot: convert wide ELO history (player columns) into long format:
    # columns -> ['game_date', 'fullName', 'ELO']
    df_unpivot = pd.melt(
    elo_history_df.reset_index(), # Reset index to keep game_date as column
    id_vars=['game_date'], # Keep 'game_date' fixed
    var_name='fullName', # Column name for player names
    value_name='ELO' # Column name for ELO values
    )

    # Only keep players whose highest ELO exceeded 1300 (reduce the computational effort and consider only above-average players)
    df_filtered = pd.merge(
    highest_elo_df[highest_elo_df['highest_ELO']>1300], # Filter above-average players
    df_unpivot,
    on='fullName', # Merge on player name
    how='left'
    )

     # Extract only relevant columns for pivoting
    elo_history_filtered_df = df_filtered[['game_date','fullName','ELO']]

    # Pivot back the table: 
    # Rows: game_date, Columns: fullName, Values: ELO
    df_history_players_pivot = elo_history_filtered_df.pivot(index='game_date', columns='fullName', values='ELO')

    # Generate bar chart races for the historic ELO values
    get_yearly_quarterly_monthly_graphs('historic', df_history_players_pivot, 'Historic NBA players')

    # Compute cumulative maximum ELO per player over time
    elo_max_df = df_history_players_pivot.cummax()

    # Generate bar chart races for maximum ELO values
    get_yearly_quarterly_monthly_graphs('highest', elo_max_df, 'Historic NBA players (MAX)')

def elo_rating_active_players(elo_history_df, highest_elo_df, df_players):
    """
    Generate bar chart race animations for active NBA players' ELO ratings (yearly, quarterly,
    and monthly animations)

    Parameters:
        elo_history_df (pd.DataFrame): ELO history with dates as index and players as columns
        highest_elo_df (pd.DataFrame): DataFrame with columns ['fullName','highest_ELO']
        df_players (pd.DataFrame): Player game stats including 'game_date' and 'fullName'

    Returns:
        None (outputs animations to 'outputs' folder)
    """
    # Unpivot: convert wide ELO history (player columns) into long format:
    # columns -> ['game_date', 'fullName', 'ELO']
    df_unpivot = pd.melt(
    elo_history_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO'
    )

    # Only keep players whose highest ELO exceeded 1300 (reduce the computational effort and consider only above-average players)
    df_filtered = pd.merge(
    highest_elo_df[highest_elo_df['highest_ELO']>1300], # Filter above-average players
    df_unpivot,
    on='fullName', # Merge on player name
    how='left'
    )

    # Retrieve first (debut) and last (retirement) game date for each player
    df_career_length = df_players.groupby('fullName')['game_date'].agg(['min', 'max']).reset_index()

    # Join ELO with career period 
    df_active_filtered = pd.merge(
    df_filtered,
    df_career_length,
    on='fullName',
    how='left'
    )

    # Keep only rows where game_date is within player's active career
    elo_active_filtered_df = df_active_filtered[(df_active_filtered['min']<=df_active_filtered['game_date'])&(df_active_filtered['game_date']<=df_active_filtered['max'])][['game_date','fullName','ELO']]

    
    
    # Pivot back the table: 
    # Rows: game_date, Columns: fullName, Values: ELO
    df_active_players_pivot = elo_active_filtered_df.pivot(index='game_date', columns='fullName', values='ELO')

    # Generate bar chart races for active players
    get_yearly_quarterly_monthly_graphs('active', df_active_players_pivot, 'Active NBA players')

def best_players_analysis(elo_history_df, team_elo_df, opponent_elo_df):
    """
    Analyze ELO ratings for a selection of historically great NBA players and
    compute game-by-game statistics including ELO change, win probability, and career progress.

    Parameters:
        elo_history_df (pd.DataFrame): ELO history per player (columns=players, index=game_date)
        team_elo_df (pd.DataFrame): Team ELO associated with each player per game
        opponent_elo_df (pd.DataFrame): Opponent team ELO associated with each player per game

    Returns:
        pd.DataFrame: Merged DataFrame with calculated statistics per game for selected top players
    """
    # ------------------ SELECT TOP PLAYERS ------------------
    selected_players = ['Kareem Abdul-Jabbar', 'Ray Allen', 'Giannis Antetokounmpo', 'Carmelo Anthony', 'Nate Archibald',
                        'Paul Arizin', 'Charles Barkley', 'Rick Barry', 'Elgin Baylor', 'Dave Bing', 'Kobe Bryant',
                        'Larry Bird', 'Wilt Chamberlain', 'Bob Cousy', 'Dave Cowens', 'Dave DeBusschere', 'Clyde Drexler',
                        'Julius Erving', 'Patrick Ewing', 'Dolph Schayes', 'George Gervin', 'Hal Greer', 'John Havlicek',
                        'Elvin Hayes', 'Allen Iverson', 'Magic Johnson', 'Sam Jones', 'Michael Jordan', 'Jason Kidd',
                        'Karl Malone', 'Pete Maravich', 'Moses Malone', 'Kevin McHale', 'George Mikan', 'Dirk Nowitzki',
                        'Hakeem Olajuwon', 'Shaquille O\'Neal', 'Robert Parish', 'Bob Pettit', 'Paul Pierce', 'Scottie Pippen',
                        'Reggie Miller', 'Dominique Wilkins', 'Dennis Rodman', 'Chris Paul', 'Dwyane Wade', 'Stephen Curry',
                        'Kevin Durant', 'Tim Duncan', 'Kevin Garnett', 'Kawhi Leonard', 'Damian Lillard', 'LeBron James',
                        'Anthony Davis', 'Russell Westbrook', 'James Harden', 'Steve Nash', 'John Stockton', 'Walt Frazier',
                        'Bill Russell', 'Bob McAdoo', 'Bill Walton', 'Wes Unseld', 'Lenny Wilkens', 'Jerry West', 'Bill Sharman',
                        'Dwight Howard', 'Tracy McGrady', 'Vince Carter', 'Tony Parker', 'Pau Gasol', 'Kyrie Irving', 'Klay Thompson',
                        'Dikembe Mutombo', 'Yao Ming', 'Vlade Divac', 'Manu Ginobili', 'David Robinson', 'Nikola Jokic', 'Oscar Robertson',
                        'Gary Payton', 'Isiah Thomas', 'Jrue Holiday', 'Luka Doncic', 'Shai Gilgeous-Alexander', 'Chauncey Billups',
                        'Jimmy Butler', 'Andre Iguodala', ]
    
    # Keep only selected players in all ELO DataFrames
    elo_history_df = elo_history_df[selected_players]
    team_elo_df = team_elo_df[selected_players]
    opponent_elo_df = opponent_elo_df[selected_players]

    # Unpivot: convert wide ELO history (player columns) into long format:
    # columns -> ['game_date', 'fullName', 'ELO']
    df_player_unpivot = pd.melt(
    elo_history_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO'
    )
    # Unpivot: convert wide team ELO history (player columns) into long format:
    # columns -> ['game_date', 'fullName', 'ELO_team']
    df_team_unpivot = pd.melt(
    team_elo_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO_team'
    )

    # Unpivot: convert wide opponent team ELO history (player columns) into long format:
    # columns -> ['game_date', 'fullName', 'ELO_opponent']
    df_opponent_unpivot = pd.melt(
    opponent_elo_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO_opponent'
    )

    # Merge team and opponent ELOs on player name and date
    df_merged = pd.merge(
    df_team_unpivot,
    df_opponent_unpivot,
    on=['fullName', 'game_date'],
    how='left'
    )

    # Merge team and opponent ELOs, and player's ELO on player name and date
    df_merged2 = pd.merge(
    df_merged,
    df_player_unpivot,
    on=['fullName', 'game_date'],
    how='left'
    )

    # Filter out rows with missing team ELO (dates in wich the player didn't play in a game)
    df_filtered = df_merged2[(~df_merged2['ELO_team'].isnull()) & (~df_merged2['ELO_team'].isna())]

    # Sort and calculate game stats
    df_sorted = df_filtered.sort_values(["fullName", "game_date"])
    df_sorted["game_date"] = pd.to_datetime(df_sorted["game_date"])

    # Compute previous game ELO
    df_sorted["ELO_previous_game"] = df_sorted.groupby("fullName")["ELO"].shift(1)
    df_sorted["ELO_previous_game"] = df_sorted["ELO_previous_game"].fillna(1000)

    # Change in ELO
    df_sorted["ELO_change"] = df_sorted["ELO"]-df_sorted["ELO_previous_game"]

    # Calculate debut date and days since debut
    df_sorted["debut_date"] = df_sorted.groupby("fullName")["game_date"].transform("min")
    df_sorted["debut_date"] = pd.to_datetime(df_sorted["debut_date"])
    df_sorted["days_since_debut"] = (df_sorted["game_date"] - df_sorted["debut_date"]).dt.days

    # Team ELO difference (team vs opponent)
    df_sorted["team_ELO_difference"] = df_sorted["ELO_team"]-df_sorted["ELO_opponent"]

    # Determine game result: Win (W), Loss (L), or No Play (NP)
    df_sorted['game_result'] = np.select(
        [
            df_sorted["ELO"] == df_sorted["ELO_previous_game"],
            df_sorted["ELO"] > df_sorted["ELO_previous_game"],
            df_sorted["ELO"] < df_sorted["ELO_previous_game"],
        ],
        ["NP", "W", "L"], default="NP"
    )

    # Calculate win probability based on team ELO difference
    df_sorted['win_probability'] = 1 / (1 + np.power(10, -df_sorted["team_ELO_difference"] / 400))

    # Merge player game stats to include minutes played and game type

    df_players, _, _ = load_players()
    df_players["game_date"] = pd.to_datetime(df_players["game_date"])

    return pd.merge(df_sorted, df_players[['fullName', 'game_date', 'gameType','minutes']], on = ['fullName', 'game_date'], how = 'left')
    
def average_elo_analysis(elo_history_df, team_elo_df):
    """
    Compute the average ELO rating per player and save the results to an Excel file.

    Parameters:
        elo_history_df (pd.DataFrame): ELO history per player (columns=players, index=game_date)
        team_elo_df (pd.DataFrame): Team ELO per player (columns=players, index=game_date)

    Returns:
        None (outputs an Excel file)
    """
    # Unpivot: convert wide ELO history (player columns) into long format:
    # columns -> ['game_date', 'fullName', 'ELO']
    df_player_unpivot = pd.melt(
    elo_history_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO'
    )
    # Unpivot: convert wide ELO history (player columns) into long format:
    # columns -> ['game_date', 'fullName', 'ELO_team']
    df_team_unpivot = pd.melt(
    team_elo_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO_team'
    )

    # Merge team and opponent ELOs on player name and date
    df_merged = pd.merge(
    df_team_unpivot,
    df_player_unpivot,
    on=['fullName', 'game_date'],
    how='left'
    )
    # Filter out rows where team ELO is missing (the player didn't play a game on that date)
    df = df_merged[(~df_merged['ELO_team'].isnull()) & (~df_merged['ELO_team'].isna())][['game_date','fullName','ELO']]

    # Compute mean ELO per player, sort in descending order, and save the results
    df_mean_elo = df.groupby('fullName')['ELO'].mean().reset_index().sort_values('ELO', ascending=False)
    df_mean_elo.to_excel(os.path.join("outputs", f"mean_elo.xlsx"), index=False)


if __name__ == "__main__":
    # ------------------ LOAD DATA ------------------
    # Load all game records into a DataFrame
    df_games = load_games()
    # Load player statistics, sorted unique dates, and list of player names
    df_players, sorted_dates, players = load_players()

    # Load previously saved ELO data if exists (elo history, last ELO, highest ELO, team and opponent team ELOs)
    elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO = load_files()

    # Uncomment the next line if you want to recompute ELO from scratch
    # elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO = elo_computation(df_games, df_players, sorted_dates, players)
    

    # ------------------ CREATE DATAFRAMES ------------------
    # Convert ELO history dictionary to DataFrame, transpose so players are columns
    elo_history_df = pd.DataFrame.from_dict(elo_history, orient='index').transpose()
    # Add a starting fictional game date at the beginning (the day before the first NBA game to handle the first 1000 ELO rating)
    elo_history_df['game_date']=[datetime.date(1946,11,25)]+sorted_dates
    # Set 'game_date' as the index for time-series operations
    elo_history_df.set_index('game_date', inplace=True)

    # Idem for team_player_ELO
    team_player_ELO_df = pd.DataFrame.from_dict(team_player_ELO, orient='index').transpose()
    team_player_ELO_df['game_date']=[datetime.date(1946,11,25)]+sorted_dates
    team_player_ELO_df.set_index('game_date', inplace=True)

    # Idem for opponent_team_player_ELO
    opponent_team_player_ELO_df = pd.DataFrame.from_dict(opponent_team_player_ELO, orient='index').transpose()
    opponent_team_player_ELO_df['game_date']=[datetime.date(1946,11,25)]+sorted_dates
    opponent_team_player_ELO_df.set_index('game_date', inplace=True)

    # ------------------ ANALYSIS ------------------
    # Compute average ELO per player and save to Excel
    average_elo_analysis(elo_history_df, team_player_ELO_df)
    # Convert highest ELO dictionary to DataFrame
    highest_ELO_df = pd.DataFrame(list(highest_ELO.items()), columns=['fullName', 'highest_ELO'])

    # Generate bar chart race animations
    elo_rating_active_players(elo_history_df, highest_ELO_df, df_players)
    elo_historic_rating_players(elo_history_df, highest_ELO_df)

    # Save the best players analysis to an Excel file
    best_players_df = best_players_analysis(elo_history_df, team_player_ELO_df, opponent_team_player_ELO_df)
    best_players_df.to_excel(os.path.join("outputs", f"best_playersv2.xlsx"), index=False)