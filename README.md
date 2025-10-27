# nba-elo
A Python tool to compute historical ELO ratings for NBA players, inspired by chess. It updates ratings from game data to track performance across eras and matchups. Compare legends head-to-head, visualize rating trajectories, and explore the endless debate of who is the true NBA GOAT.

The data is sourced from [Kaggle-NBA Dataset](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores), covering NBA games from the league's inception up to the **end of the 2024-2025 season**. The methodology is fully explained in a didactic manner on the [project website](https://jcllanu.github.io/nba-elo/), with **multiple examples, interactive charts, and videos** showing how player ELO ratings are calculated and how they evolve over time. The website also features a discussion on who the NBA GOAT is according to this metric.

## Project Structure

- `inputs/` → CSV files containing game data and player statistics (the latter is too large to include directly, but can be downloaded from the indicated source).  
- `outputs/` → Results including charts, videos, Excel exports, and working files.  
- `data/` → Intermediate ELO files for resuming calculations without recomputing everything (elo_history_until_2025-06-22.csv was too large and hasn't been uploaded).  
- `nba_elo.py` → Main Python code to load data, compute ELO, and generate analyses and visualizations.
