import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm.auto import tqdm
import os
import bar_chart_race as bcr
import imageio_ffmpeg as ffmpeg
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = ffmpeg.get_ffmpeg_exe()

def load_games():
    df_games = pd.read_csv('Games.csv')
    df_games['home_team'] = df_games['hometeamCity'].str.strip() + ' ' + df_games['hometeamName'].str.strip()
    df_games['away_team'] = df_games['awayteamCity'].str.strip() + ' ' + df_games['awayteamName'].str.strip()
    df_games['gameDate'] = pd.to_datetime(df_games['gameDate'])
    df_games['game_date'] = df_games['gameDate'].dt.date
    df_games=df_games[['game_date', 'home_team', 'away_team', 'homeScore', 'awayScore', 'gameType']]
    return df_games

def load_players():
    df_players = pd.read_csv('PlayerStatistics.csv')
    df_players['gameDate'] = pd.to_datetime(df_players['gameDate'])
    df_players['game_date'] = df_players['gameDate'].dt.date
    sorted_dates = sorted(df_players['game_date'].unique())
    df_players['fullName'] = df_players['firstName'].str.strip() + ' ' + df_players['lastName'].str.strip()
    df_players['player_team'] = df_players['playerteamCity'].str.strip() + ' ' + df_players['playerteamName'].str.strip()
    df_players['opponent_team'] = df_players['opponentteamCity'].str.strip() + ' ' + df_players['opponentteamName'].str.strip()
    df_players['numMinutes'] = df_players['numMinutes'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minutes = df_players['numMinutes'].astype(int)
    seconds = ((df_players['numMinutes'] - minutes) * 100).round()
    df_players['minutes']= minutes+seconds/60
    players = df_players['fullName'].unique()
    df_players = df_players[['game_date','fullName','player_team', 'opponent_team', 'minutes', 'gameId', 'gameType']]
    return df_players, sorted_dates, players

def save(last_date, elo_history, last_ELO, highest_ELO, team_ELO, opponent_team_ELO):
    save_dir = 'data'
    # Save full ELO history
    elo_history_df = pd.DataFrame.from_dict(elo_history, orient='index').transpose()
    elo_history_df.to_csv(os.path.join(save_dir, f"elo_history_until_{last_date}.csv"), index=False)

    # Save last ELO per player
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

        

# Function to compute team average ELO based on minutes played
def calculate_team_ELO(players_rows, last_ELO):
    total_minutes = players_rows['minutes'].sum()
    num_players = len(players_rows)
    # Handle invalid or missing minutes
    assert num_players > 0, f"{players_rows}"

    if total_minutes <= 0 or num_players < 5:
        # Default total minutes (full game for 5 players)
        total_minutes = 48 * 5
        minutes_per_player = total_minutes / num_players
        weighted_sum = sum(last_ELO[row['fullName']] * minutes_per_player for _, row in players_rows.iterrows())
        return weighted_sum / total_minutes, total_minutes, minutes_per_player
    else:
        weighted_sum = sum(last_ELO[row['fullName']] * row['minutes'] for _, row in players_rows.iterrows())
        return weighted_sum / total_minutes, total_minutes, None

def elo_computation(df_games, df_players, sorted_dates, players):
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)
    # --- LOAD CHECKPOINT IF EXISTS ---
    saved_files = [f for f in os.listdir(save_dir) if f.startswith("elo_history_until_")]
    if saved_files:
        # Find last saved date
        latest_file = sorted(saved_files)[-1]
        last_saved_date = latest_file.split("_until_")[1].split(".csv")[0]

        print(f"🔁 Resuming from {last_saved_date}...")

        # Load ELO history
        elo_history_df = pd.read_csv(os.path.join(save_dir, latest_file))
        elo_history = {player: elo_history_df[player].dropna().tolist() for player in elo_history_df.columns}
        length = len(elo_history_df)

        # Load last ELO
        last_elo_df = pd.read_csv(os.path.join(save_dir, f"last_elo_until_{last_saved_date}.csv"))
        last_ELO = dict(zip(last_elo_df['player'], last_elo_df['last_ELO']))

        # Load highest ELO
        highest_elo_df = pd.read_csv(os.path.join(save_dir, f"highest_elo_until_{last_saved_date}.csv"))
        highest_ELO = dict(zip(highest_elo_df['player'], highest_elo_df['highest_ELO']))

        # Load team ELO
        team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"team_elo_until_{last_saved_date}.csv"))
        team_player_ELO = {player: team_player_ELO_df[player].tolist() for player in team_player_ELO_df.columns}

        # Load opponent team ELO
        opponent_team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"opponent_team_until_{last_saved_date}.csv"))
        opponent_team_player_ELO = {player: opponent_team_player_ELO_df[player].tolist() for player in opponent_team_player_ELO_df.columns}

        # Load players
        players = list(last_ELO.keys())

        # Filter later dates
        sorted_dates_filtered = [d for d in sorted_dates if d > datetime.datetime.strptime(last_saved_date, "%Y-%m-%d").date()]
        df_players_filtered = df_players[df_players['game_date']> datetime.datetime.strptime(last_saved_date, "%Y-%m-%d").date()].copy()
        df_games_filtered = df_games[df_games['game_date']> datetime.datetime.strptime(last_saved_date, "%Y-%m-%d").date()].copy()
        game_date = last_saved_date
    else:
        print("🚀 Initializing from scratch...")

        # Initialize player ELO structures
        elo_history = {player: [1000] for player in players} # Full ELO history per player
        last_ELO = {player: 1000 for player in players} # Latest ELO per player
        highest_ELO = {player: 1000 for player in players} # Highest ELO achieved per player
        team_player_ELO = {player: [None] for player in players} # Team ELO of the player's team
        opponent_team_player_ELO = {player: [None] for player in players} # Team ELO of the player's opponent team
        length = 1 # Track length of ELO history
        df_players_filtered=df_players.copy()
        df_games_filtered=df_games.copy()
        sorted_dates_filtered = sorted_dates

    df_games_filtered.set_index('game_date', inplace=True)
    df_players_filtered.set_index('game_date', inplace=True)

    

    # Main loop through all game dates
    for game_date in tqdm(sorted_dates_filtered):
        daily_games = df_games_filtered.loc[[game_date]]
        players_today = df_players_filtered.loc[[game_date]]

        for game in daily_games.itertuples(index=False):
            home_team = game.home_team
            away_team = game.away_team
            team_players = {
                home_team: players_today[players_today['player_team'] == home_team],
                away_team: players_today[players_today['player_team'] == away_team]
            }
            team_ELO = {}
            team_played_time = {}
            default_minutes = {}

            # Calculate each team's average ELO and total minutes played
            for team in [home_team, away_team]:
                players_rows = team_players[team]
                team_ELO[team], team_played_time[team], default_minutes[team] = calculate_team_ELO(players_rows, last_ELO)

            # Calculate expected win probability using ELO difference
            Q_home = 1 / (1 + 10 ** ((team_ELO[away_team] - team_ELO[home_team]) / 400))
            Q_away = 1 - Q_home
            team_Q = {home_team: Q_home, away_team: Q_away}

            # Determine K factor (higher for playoff games)
            K = 32 if game.gameType == 'Playoffs' else 16

            # Update ELO ratings for each player based on game result
            for team in [home_team, away_team]:
                won = (team == home_team and game.homeScore > game.awayScore) or \
                    (team == away_team and game.awayScore > game.homeScore)
                S = 1 if won else 0
                E = team_Q[team]

                players_rows = team_players[team]

                for row in players_rows.itertuples(index=False):
                    player = row.fullName
                    current_ELO = last_ELO[player]
                    # Use default minutes if available
                    minutes_played = row.minutes if default_minutes[team] is None else default_minutes[team]

                    # Update ELO based on player's minutes and game outcome (per 36 min)
                    delta =  (minutes_played / 36) * K * (S - E)
                    # print(f"Player: {player}. Minutes played {minutes_played}. Total time {team_time[team]} Delta {delta}")
                    new_ELO = current_ELO + delta

                    # Save new ELO
                    elo_history[player].append(new_ELO)
                    last_ELO[player] = new_ELO
                    highest_ELO[player] = max(highest_ELO[player], new_ELO)
                    team_player_ELO[player].append(team_ELO[home_team] if team == home_team else team_ELO[away_team])
                    opponent_team_player_ELO[player].append(team_ELO[away_team] if team == home_team else team_ELO[home_team])
                    


        df_games_filtered.drop(index=game_date, inplace=True, errors='ignore')
        df_players_filtered.drop(index=game_date, inplace=True, errors='ignore')

        # Ensure every player has the same number of ELO entries (carry forward)
        length += 1
        for player in elo_history:
            if len(elo_history[player]) < length:
                elo_history[player].append(last_ELO[player])
                team_player_ELO[player].append(None)
                opponent_team_player_ELO[player].append(None)


        # Show progress every 100 dates
        if length % 100 == 0:
            print(f"\n📅 Date: {game_date}")

            # Top 10 sorted lists
            top_highest = sorted(highest_ELO.items(), key=lambda x: x[1], reverse=True)[:10]
            top_current = sorted(last_ELO.items(), key=lambda x: x[1], reverse=True)[:10]

            # Header with fixed-width columns
            name_width = 20  # Adjust as needed
            print(
                f"{'🏆 Highest Player ELO 🏆':<{name_width}}        "
                f"{'🔥 Current Player ELO 🔥':<{name_width}}"
            )

            # Rows
            for (high_name, high_elo), (curr_name, curr_elo) in zip(top_highest, top_current):
                print(
                    f"{high_name:<{name_width}} {high_elo:>7.2f}    "
                    f"{curr_name:<{name_width}} {curr_elo:>7.2f}"
                )

            # Save progress every 500 dates
            if length % 1000 == 0:
                print(f"\n💾 Saving progress at date {game_date}...")
                save(game_date, elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO)


    print(f"\n💾 Saving progress at date {game_date}...")
    save(game_date, elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO)
    return elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO

def get_yearly_quarterly_monthly_graphs(mode, df, title):
    df.index = pd.to_datetime(df.index) # Date format

    # YEAR
    df_active_players_year_avg = df.resample('Y').mean() # Get annual sample aggregating by mean
    df_active_players_year_avg.dropna(how='all', inplace=True)
    bcr.bar_chart_race(
        df=df_active_players_year_avg.fillna(0),
        filename='outputs\\yearly_' + mode + '_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.mp4',      
        orientation='h',               # horizontal bars
        sort='desc',
        n_bars=15,
        fixed_order=False,
        fixed_max=True,
        steps_per_period=20,
        period_length=1000,
        title='Yearly ELO Rating - ' + title,
        interpolate_period=False,
        bar_size=.8,
        cmap='dark12',
        dpi=144,
        period_label={'x': 0.99, 'y': 0.1, 'ha': 'right', 'va': 'center'},
    )

    # QUARTER
    df_active_players_quarter_avg = df.resample('Q').mean() # Get quarterly sample aggregating by mean 
    df_active_players_quarter_avg.dropna(how='all', inplace=True)
    bcr.bar_chart_race(
        df=df_active_players_quarter_avg.fillna(0),
        filename='outputs\\quarterly_' + mode + '_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.mp4',      
        orientation='h',               # horizontal bars
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

    # MONTH
    df_active_players_month_avg = df.resample('M').mean() # Get monthly sample aggregating by mean
    df_active_players_month_avg.dropna(how='all', inplace=True)
    bcr.bar_chart_race(
        df=df_active_players_month_avg.fillna(0),
        filename='outputs\\monthly_' + mode + '_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.mp4',      
        orientation='h',               # horizontal bars
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
    # Unpivot table (3 columns): date, player name and ELO (of that player in that date)
    df_unpivot = pd.melt(
    elo_history_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO'
    )
    # Filter those with highest_ELO > 1300
    df_filtered = pd.merge(
    highest_elo_df[highest_elo_df['highest_ELO']>1300],
    df_unpivot,
    on='fullName',
    how='left'
    )
    # Remove highest ELO
    elo_history_filtered_df = df_filtered[['game_date','fullName','ELO']]
    # Pivot the table 
    df_history_players_pivot = elo_history_filtered_df.pivot(index='game_date', columns='fullName', values='ELO')

    get_yearly_quarterly_monthly_graphs('historic', df_history_players_pivot, 'Historic NBA players')

    elo_max_df = df_history_players_pivot.cummax()

    get_yearly_quarterly_monthly_graphs('highest', elo_max_df, 'Historic NBA players (MAX)')

    

def elo_rating_active_players(elo_history_df, highest_elo_df, df_players):
    # Unpivot table (3 columns): date, player name and ELO (of that player in that date)
    df_unpivot = pd.melt(
    elo_history_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO'
    )
    # Filter those with highest_ELO > 1300
    df_filtered = pd.merge(
    highest_elo_df[highest_elo_df['highest_ELO']>1300],
    df_unpivot,
    on='fullName',
    how='left'
    )

    # Retrieve debut and retirement date of each player
    df_career_length = df_players.groupby('fullName')['game_date'].agg(['min', 'max']).reset_index()

    # Join 
    df_active_filtered = pd.merge(
    df_filtered,
    df_career_length,
    on='fullName',
    how='left'
    )

    # Filter the dates when the player is active
    elo_active_filtered_df = df_active_filtered[(df_active_filtered['min']<=df_active_filtered['game_date'])&(df_active_filtered['game_date']<=df_active_filtered['max'])][['game_date','fullName','ELO']]

    
    
    # Pivot the table 
    df_active_players_pivot = elo_active_filtered_df.pivot(index='game_date', columns='fullName', values='ELO')


    get_yearly_quarterly_monthly_graphs('active', df_active_players_pivot, 'Active NBA players')


def best_players_analysis(elo_history_df, team_elo_df, opponent_elo_df):
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

    elo_history_df = elo_history_df[selected_players]
    team_elo_df = team_elo_df[selected_players]
    opponent_elo_df = opponent_elo_df[selected_players]

    # Unpivot table (3 columns): date, player name and ELO (of that player in that date)
    df_player_unpivot = pd.melt(
    elo_history_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO'
    )
    # Unpivot table (3 columns): date, player name and ELO (of that player in that date)
    df_team_unpivot = pd.melt(
    team_elo_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO_team'
    )

    # Unpivot table (3 columns): date, player name and ELO (of that player in that date)
    df_opponent_unpivot = pd.melt(
    opponent_elo_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO_opponent'
    )

    df_merged = pd.merge(
    df_team_unpivot,
    df_opponent_unpivot,
    on=['fullName', 'game_date'],
    how='left'
    )

    df_merged2 = pd.merge(
    df_merged,
    df_player_unpivot,
    on=['fullName', 'game_date'],
    how='left'
    )
    df_filtered = df_merged2[(~df_merged2['ELO_team'].isnull()) & (~df_merged2['ELO_team'].isna())]
    df_sorted = df_filtered.sort_values(["fullName", "game_date"])
    df_sorted["game_date"] = pd.to_datetime(df_sorted["game_date"])
    df_sorted["ELO_previous_game"] = df_sorted.groupby("fullName")["ELO"].shift(1)
    df_sorted["ELO_previous_game"] = df_sorted["ELO_previous_game"].fillna(1000)
    df_sorted["ELO_change"] = df_sorted["ELO"]-df_sorted["ELO_previous_game"]
    df_sorted["debut_date"] = df_sorted.groupby("fullName")["game_date"].transform("min")
    df_sorted["debut_date"] = pd.to_datetime(df_sorted["debut_date"])
    df_sorted["days_since_debut"] = (df_sorted["game_date"] - df_sorted["debut_date"]).dt.days
    df_sorted["team_ELO_difference"] = df_sorted["ELO_team"]-df_sorted["ELO_opponent"]
    df_sorted['game_result'] = np.select(
        [
            df_sorted["ELO"] == df_sorted["ELO_previous_game"],
            df_sorted["ELO"] > df_sorted["ELO_previous_game"],
            df_sorted["ELO"] < df_sorted["ELO_previous_game"],
        ],
        ["NP", "W", "L"], default="NP"
    )
    df_sorted['win_probability'] = 1 / (1 + np.power(10, -df_sorted["team_ELO_difference"] / 400))

    df_players, _, _ =load_players()
    df_players["game_date"] = pd.to_datetime(df_players["game_date"])

    return pd.merge(df_sorted, df_players[['fullName', 'game_date', 'gameType','minutes']], on = ['fullName', 'game_date'], how = 'left')



    







    

def load_files():
    save_dir = 'data'
    saved_files = [f for f in os.listdir(save_dir) if f.startswith("elo_history_until_")]
    if saved_files:
        # Find last saved date
        latest_file = sorted(saved_files)[-1]
        last_saved_date = latest_file.split("_until_")[1].split(".csv")[0]

        print(f"🔁 Resuming from {last_saved_date}...")

        # Load ELO history
        elo_history_df = pd.read_csv(os.path.join(save_dir, latest_file))
        elo_history = {player: elo_history_df[player].dropna().tolist() for player in elo_history_df.columns}
        length = len(elo_history_df)

        # Load last ELO
        last_elo_df = pd.read_csv(os.path.join(save_dir, f"last_elo_until_{last_saved_date}.csv"))
        last_ELO = dict(zip(last_elo_df['player'], last_elo_df['last_ELO']))

        # Load highest ELO
        highest_elo_df = pd.read_csv(os.path.join(save_dir, f"highest_elo_until_{last_saved_date}.csv"))
        highest_ELO = dict(zip(highest_elo_df['player'], highest_elo_df['highest_ELO']))

        # Load team ELO
        team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"team_elo_until_{last_saved_date}.csv"))
        team_player_ELO = {player: team_player_ELO_df[player].tolist() for player in team_player_ELO_df.columns}

        # Load opponent team ELO
        opponent_team_player_ELO_df = pd.read_csv(os.path.join(save_dir, f"opponent_team_until_{last_saved_date}.csv"))
        opponent_team_player_ELO = {player: opponent_team_player_ELO_df[player].tolist() for player in opponent_team_player_ELO_df.columns}

        return elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO
    
def average_elo_analysis(elo_history_df, team_elo_df):

    # Unpivot table (3 columns): date, player name and ELO (of that player in that date)
    df_player_unpivot = pd.melt(
    elo_history_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO'
    )
    # Unpivot table (3 columns): date, player name and ELO (of that player in that date)
    df_team_unpivot = pd.melt(
    team_elo_df.reset_index(),
    id_vars=['game_date'],
    var_name='fullName',
    value_name='ELO_team'
    )

    df_merged = pd.merge(
    df_team_unpivot,
    df_player_unpivot,
    on=['fullName', 'game_date'],
    how='left'
    )
    df = df_merged[(~df_merged['ELO_team'].isnull()) & (~df_merged['ELO_team'].isna())][['game_date','fullName','ELO']]
    df_mean_elo = df.groupby('fullName')['ELO'].mean().reset_index().sort_values('ELO', ascending=False)
    df_mean_elo.to_excel(os.path.join("outputs", f"mean_elo.xlsx"), index=False)

df_games = load_games()
df_players, sorted_dates, players = load_players()
elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO = load_files()
# elo_history, last_ELO, highest_ELO, team_player_ELO, opponent_team_player_ELO = elo_computation(df_games, df_players, sorted_dates, players)

elo_history_df = pd.DataFrame.from_dict(elo_history, orient='index').transpose()
elo_history_df['game_date']=[datetime.date(1946,11,25)]+sorted_dates
elo_history_df.set_index('game_date', inplace=True)

 
# print(elo_history_df)
team_player_ELO_df = pd.DataFrame.from_dict(team_player_ELO, orient='index').transpose()
team_player_ELO_df['game_date']=[datetime.date(1946,11,25)]+sorted_dates
team_player_ELO_df.set_index('game_date', inplace=True)
# print(team_player_ELO_df)
opponent_team_player_ELO_df = pd.DataFrame.from_dict(opponent_team_player_ELO, orient='index').transpose()
opponent_team_player_ELO_df['game_date']=[datetime.date(1946,11,25)]+sorted_dates
opponent_team_player_ELO_df.set_index('game_date', inplace=True)
# print(opponent_team_player_ELO_df)

# average_elo_analysis(elo_history_df, team_player_ELO_df)

# highest_ELO_df = pd.DataFrame(list(highest_ELO.items()), columns=['fullName', 'highest_ELO'])
# elo_rating_active_players(elo_history_df, highest_ELO_df, df_players)
# elo_historic_rating_players(elo_history_df, highest_ELO_df)

best_players_df = best_players_analysis(elo_history_df, team_player_ELO_df, opponent_team_player_ELO_df)

best_players_df.to_excel(os.path.join("outputs", f"best_playersv2.xlsx"), index=False)