import requests
import chess
import chess.pgn
import chess.engine
from collections import defaultdict
from io import StringIO
import os
from datetime import datetime
import pickle
import importlib
import json
import sys

# Load config from command-line argument
if len(sys.argv) < 2:
    raise ValueError("Usage: python script.py path/to/config.py")
config_path = sys.argv[1]
config_dir = os.path.dirname(config_path)
sys.path.append(config_dir)
config_module_name = os.path.splitext(os.path.basename(config_path))[0]
config = importlib.import_module(config_module_name)

# Extract variables from config
LICHEss_USERNAME = config.LICHEss_USERNAME
CHESS_COM_USERNAME = config.CHESS_COM_USERNAME
USER_AGENT = config.USER_AGENT
NUM_GAMES = config.NUM_GAMES
MIN_OCCURRENCES = config.MIN_OCCURRENCES
MIN_PLY = config.MIN_PLY
EVAL_THRESHOLD = config.EVAL_THRESHOLD
STOCKFISH_PATH = config.STOCKFISH_PATH
ANALYSIS_TIME = config.ANALYSIS_TIME
THREADS = config.THREADS
HASH_MB = config.HASH_MB

# Dynamic file prefixes and base directory
prefix = LICHEss_USERNAME or CHESS_COM_USERNAME or "default_user"
base_dir = f"tmp/{prefix}"
os.makedirs(base_dir, exist_ok=True)

LICHEss_LOCAL_PGN = os.path.join(base_dir, f"{prefix}_lichess_games.pgn") if LICHEss_USERNAME else None
CHESS_LOCAL_PGN = os.path.join(base_dir, f"{prefix}_chess_games.pgn") if CHESS_COM_USERNAME else None
COMMON_PICKLE = os.path.join(base_dir, f"{prefix}_common_positions.pkl")  # Shared, as positions are combined
PATH_PICKLE = os.path.join(base_dir, f"{prefix}_paths.pkl")  # Shared
GAME_IDS_PICKLE = os.path.join(base_dir, f"{prefix}_game_ids.pkl")  # Shared, as IDs are unique per site
ANALYSIS_PGN = os.path.join(base_dir, f"{prefix}_analysis.pgn")

# Helper function to count games in PGN
def count_games_in_pgn(pgn_text):
    if not pgn_text.strip():
        return 0
    pgn_io = StringIO(pgn_text)
    count = 0
    while chess.pgn.read_game(pgn_io) is not None:
        count += 1
    return count

def count_games_in_file(pgn_file):
    if not pgn_file or not os.path.exists(pgn_file):
        return 0
    with open(pgn_file, 'r') as f:
        pgn_text = f.read()
    return count_games_in_pgn(pgn_text)

# Function to process games and update common positions
def process_games(pgn_text, common_positions, fen_to_path):
    if not pgn_text.strip():
        return
    pgn_io = StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        # Determine user's color
        white = game.headers.get("White", "").lower()
        black = game.headers.get("Black", "").lower()
        user_lower = (LICHEss_USERNAME or CHESS_COM_USERNAME).lower()
        if user_lower == white:
            color = chess.WHITE
        elif user_lower == black:
            color = chess.BLACK
        else:
            continue  # Skip if not user's game
        board = game.board()
        ply = 0
        path = []
        for node in game.mainline():
            if ply >= MIN_PLY and board.turn == color:
                fen = board.fen()
                move = node.move
                uci = move.uci()
                common_positions[fen][uci] += 1
                if fen not in fen_to_path:
                    fen_to_path[fen] = path[:]
            next_move_san = board.san(node.move)
            board.push(node.move)
            path.append(next_move_san)
            ply += 1

# Load cached common positions if exists
common_positions = defaultdict(lambda: defaultdict(int))
if os.path.exists(COMMON_PICKLE):
    try:
        with open(COMMON_PICKLE, 'rb') as f:
            converted = pickle.load(f)
        common_positions = defaultdict(lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in converted.items()})
    except (EOFError, pickle.UnpicklingError):
        print(f"Warning: Failed to load {COMMON_PICKLE} (empty or corrupted file). Initializing empty.")
        # Optionally, os.remove(COMMON_PICKLE) to delete the bad file

# Load cached paths if exists
fen_to_path = {}
if os.path.exists(PATH_PICKLE):
    try:
        with open(PATH_PICKLE, 'rb') as f:
            fen_to_path = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        print(f"Warning: Failed to load {PATH_PICKLE} (empty or corrupted file). Initializing empty.")
        # Optionally, os.remove(PATH_PICKLE) to delete the bad file

# Load cached game IDs if exists
game_ids = set()
if os.path.exists(GAME_IDS_PICKLE):
    try:
        with open(GAME_IDS_PICKLE, 'rb') as f:
            game_ids = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        print(f"Warning: Failed to load {GAME_IDS_PICKLE} (empty or corrupted file). Initializing empty.")

# Build game_ids from both local PGN if not loaded
if not game_ids:
    for local_pgn in [LICHEss_LOCAL_PGN, CHESS_LOCAL_PGN]:
        if local_pgn and os.path.exists(local_pgn):
            with open(local_pgn, 'r') as f:
                existing_pgn = f.read()
            pgn_io = StringIO(existing_pgn)
            while True:
                game = chess.pgn.read_game(pgn_io)
                if game is None:
                    break
                site = game.headers.get("Site", "")
                if site:
                    game_id = site.split("/")[-1]  # Extract GAMEID from https://lichess.org/GAMEID or chess.com equivalent
                    game_ids.add(game_id)

# Lichess fetch logic
filtered_lichess_new_pgn = ""
lichess_new_count = 0
lichess_total = 0
if LICHEss_USERNAME:
    lichess_since = None
    if os.path.exists(LICHEss_LOCAL_PGN):
        with open(LICHEss_LOCAL_PGN, 'r') as f:
            existing_pgn = f.read()
        pgn_io = StringIO(existing_pgn)
        max_utc = 0
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            utc_date = game.headers.get("UTCDate", "")
            utc_time = game.headers.get("UTCTime", "")
            if utc_date and utc_time:
                try:
                    dt = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S")
                    utc_millis = int(dt.timestamp() * 1000)
                    if utc_millis > max_utc:
                        max_utc = utc_millis
                except ValueError:
                    pass
        if max_utc > 0:
            lichess_since = max_utc + 1

    url = f"https://lichess.org/api/games/user/{LICHEss_USERNAME}"
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": USER_AGENT}
    params = {}
    if lichess_since is not None:
        params["since"] = lichess_since
    else:
        params["max"] = NUM_GAMES
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch Lichess games: {response.text}")

    lichess_new_pgn = response.text

    # Filter Lichess new_pgn for duplicates
    filtered_lichess_new_pgn = ""
    if lichess_new_pgn.strip():
        pgn_io = StringIO(lichess_new_pgn)
        new_game_ids = set()
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            site = game.headers.get("Site", "")
            if site:
                game_id = site.split("/")[-1]
                if game_id not in game_ids:
                    new_game_ids.add(game_id)
                    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                    game_pgn = game.accept(exporter)
                    if filtered_lichess_new_pgn:
                        filtered_lichess_new_pgn += "\n\n"
                    filtered_lichess_new_pgn += game_pgn
        game_ids.update(new_game_ids)

    # Count new Lichess games
    lichess_new_count = count_games_in_pgn(filtered_lichess_new_pgn)

    # Process filtered Lichess new games
    process_games(filtered_lichess_new_pgn, common_positions, fen_to_path)

    # Append to Lichess local PGN
    if filtered_lichess_new_pgn:
        mode = 'a' if os.path.exists(LICHEss_LOCAL_PGN) else 'w'
        with open(LICHEss_LOCAL_PGN, mode) as f:
            if mode == 'a':
                f.write('\n\n')
            f.write(filtered_lichess_new_pgn)

    # Count total Lichess games
    lichess_total = count_games_in_file(LICHEss_LOCAL_PGN)
    print(f"Downloaded {lichess_new_count} new games from Lichess. Total stored: {lichess_total}")
else:
    print("Lichess username not set; skipping Lichess. Total stored: 0")

# Chess.com fetch logic
filtered_chess_new_pgn = ""
chess_new_count = 0
chess_total = 0
if CHESS_COM_USERNAME:
    archives_url = f"https://api.chess.com/pub/player/{CHESS_COM_USERNAME}/games/archives"
    headers_chess = {"User-Agent": USER_AGENT}
    response = requests.get(archives_url, headers=headers_chess)
    if response.status_code == 200:
        archives = response.json().get("archives", [])
        archives.sort(reverse=True)  # Latest first
    else:
        print(f"Failed to fetch chess.com archives: {response.text}")
        archives = []

    chess_since = None
    if os.path.exists(CHESS_LOCAL_PGN):
        with open(CHESS_LOCAL_PGN, 'r') as f:
            existing_pgn = f.read()
        pgn_io = StringIO(existing_pgn)
        max_utc = 0
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            utc_date = game.headers.get("UTCDate", "")
            utc_time = game.headers.get("UTCTime", "")
            if utc_date and utc_time:
                try:
                    dt = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S")
                    utc_millis = int(dt.timestamp() * 1000)
                    if utc_millis > max_utc:
                        max_utc = utc_millis
                except ValueError:
                    pass
        if max_utc > 0:
            chess_since = max_utc + 1

    fetch_count = 0
    new_chess_pgn = ""
    for archive_url in archives:
        response = requests.get(archive_url, headers=headers_chess)
        if response.status_code == 200:
            month_games = response.json().get("games", [])
            for game_data in month_games:
                # Convert to PGN format (chess.com returns JSON, need to extract PGN)
                pgn = game_data.get("pgn", "")
                if pgn:
                    include = True
                    # Parse headers from PGN to check timestamp
                    game = chess.pgn.read_game(StringIO(pgn))
                    utc_date = game.headers.get("UTCDate", "")
                    utc_time = game.headers.get("UTCTime", "")
                    if chess_since and utc_date and utc_time:
                        try:
                            dt = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S")
                            utc_millis = int(dt.timestamp() * 1000)
                            if utc_millis < chess_since:
                                include = False
                        except ValueError:
                            include = False
                    if include:
                        if new_chess_pgn:
                            new_chess_pgn += "\n\n"
                        new_chess_pgn += pgn
                        fetch_count += 1
                if chess_since is None and fetch_count >= NUM_GAMES:
                    break
            if chess_since is None and fetch_count >= NUM_GAMES:
                break

    # Filter for duplicates
    if new_chess_pgn.strip():
        pgn_io = StringIO(new_chess_pgn)
        new_game_ids = set()
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            site = game.headers.get("Site", "")
            if site:
                game_id = site.split("/")[-1]
                if game_id not in game_ids:
                    new_game_ids.add(game_id)
                    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                    game_pgn = game.accept(exporter)
                    if filtered_chess_new_pgn:
                        filtered_chess_new_pgn += "\n\n"
                    filtered_chess_new_pgn += game_pgn
        game_ids.update(new_game_ids)

    # Count new chess.com games
    chess_new_count = count_games_in_pgn(filtered_chess_new_pgn)

    # Process filtered chess.com new games
    process_games(filtered_chess_new_pgn, common_positions, fen_to_path)

    # Append to chess.com local PGN
    if filtered_chess_new_pgn:
        mode = 'a' if os.path.exists(CHESS_LOCAL_PGN) else 'w'
        with open(CHESS_LOCAL_PGN, mode) as f:
            if mode == 'a':
                f.write('\n\n')
            f.write(filtered_chess_new_pgn)

    # Count total chess.com games
    chess_total = count_games_in_file(CHESS_LOCAL_PGN)
    print(f"Downloaded {chess_new_count} new games from chess.com. Total stored: {chess_total}")
else:
    print("chess.com username not set; skipping chess.com. Total stored: 0")

# Save updated common positions
converted = {k: dict(v) for k, v in common_positions.items()}
with open(COMMON_PICKLE, 'wb') as f:
    pickle.dump(converted, f)

# Save updated paths
with open(PATH_PICKLE, 'wb') as f:
    pickle.dump(fen_to_path, f)

# Save updated game IDs
with open(GAME_IDS_PICKLE, 'wb') as f:
    pickle.dump(game_ids, f)

print("Starting Stockfish analysis...")

# Initialize Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({"Threads": THREADS, "Hash": HASH_MB})

# List to hold analysis games for PGN
analysis_games = []

# Count total qualifying positions
total_qualifying = sum(1 for move_counts in common_positions.values() if sum(move_counts.values()) >= MIN_OCCURRENCES)
print(f"Total qualifying positions to analyze: {total_qualifying}")

# Analyze common positions
current = 0
for fen, move_counts in common_positions.items():
    total_occurrences = sum(move_counts.values())
    if total_occurrences < MIN_OCCURRENCES:
        continue

    current += 1
    percentage = (current / total_qualifying * 100) if total_qualifying > 0 else 0
    print(f"\nAnalyzing position {current}/{total_qualifying} ({percentage:.2f}%)")

    board = chess.Board(fen)
    user_is_white = board.turn == chess.WHITE
    print(f"User playing as {'White' if user_is_white else 'Black'}")

    limit = chess.engine.Limit(time=ANALYSIS_TIME)

    # Get best move and eval from White's perspective
    info = engine.analyse(board, limit)
    best_pv = info.get("pv", [])
    best_move = best_pv[0] if best_pv else None
    best_move_uci = best_move.uci() if best_move else "N/A"
    best_cp = info["score"].white().score(mate_score=100000)  # From White's perspective
    best_eval = best_cp / 100.0 if best_cp is not None else 'N/A'

    print(f"Common position (occurred {total_occurrences} times): {fen}")
    print(f"Best move: {best_move_uci}, Eval: {best_eval} pawns (White's perspective)")

    # Check if all played moves were the best move; if so, skip this position
    if len(move_counts) == 1 and next(iter(move_counts)) == best_move_uci:
        print("All plays were the best move; skipping position.")
        continue

    # Build PGN game for this position, starting from initial with path
    analysis_game = chess.pgn.Game()
    analysis_game.headers["Event"] = "Common Position Analysis"
    analysis_game.headers["Site"] = "Local Analysis"
    analysis_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")

    # Set player names based on turn (user's turn)
    user_name = LICHEss_USERNAME or CHESS_COM_USERNAME
    if board.turn == chess.WHITE:
        analysis_game.headers["White"] = user_name
        analysis_game.headers["Black"] = "Opponent"
    else:
        analysis_game.headers["White"] = "Opponent"
        analysis_game.headers["Black"] = user_name

    # Add prefix moves from path
    path = fen_to_path.get(fen, [])
    branch_node = analysis_game
    temp_board = chess.Board()
    for san in path:
        move = temp_board.parse_san(san)
        branch_node = branch_node.add_variation(move)
        temp_board.push(move)

    # Now branch_node is the node after the last prefix move (position at fen)
    # Add position comment
    branch_node.comment = f"Position occurred {total_occurrences} times. Initial eval: {best_eval} pawns (White's perspective)."

    # Add main line: best PV
    count_best = move_counts.get(best_move_uci, 0)
    if best_move:
        first_main_node = branch_node.add_variation(best_move)
        first_main_node.comment = f"Best move, played {count_best} times, Eval: {best_eval} pawns (White's perspective)"
        current_node = first_main_node
        for move in best_pv[1:5]:
            current_node = current_node.add_variation(move)

    # Analyze each played move and add as variations; track if any mistake
    has_mistake = False
    for played_uci, count in move_counts.items():
        board.set_fen(fen)  # Reset board
        played_move = chess.Move.from_uci(played_uci)
        board.push(played_move)

        info_after = engine.analyse(board, limit)
        after_cp = info_after["score"].white().score(mate_score=100000)  # From White's perspective

        loss_cp = (best_cp - after_cp) if user_is_white else (after_cp - best_cp)
        after_eval = after_cp / 100.0 if after_cp is not None else 'N/A'
        loss_pawns = loss_cp / 100.0

        if loss_cp > EVAL_THRESHOLD:
            has_mistake = True
            print(f"  - Common mistake: Played {played_uci} ({count} times), Eval loss: {loss_pawns} pawns")
        else:
            print(f"  - Played {played_uci} ({count} times), Eval loss: {loss_pawns} pawns (within threshold)")

        # Add as variation if not the best move
        if played_uci != best_move_uci:
            var_node = branch_node.add_variation(played_move)
            var_node.comment = f"Played {count} times. Eval after: {after_eval} pawns (White's perspective, loss: {loss_pawns} pawns)"

    # Only append if there was at least one mistake
    if has_mistake:
        analysis_games.append(analysis_game)
    else:
        print("No common mistakes in this position; skipping inclusion in output.")

print("All positions analyzed. Quitting engine...")

# Quit engine
engine.quit()

print("Engine quit. Exporting analysis to PGN...")

# Export analysis to PGN file
with open(ANALYSIS_PGN, 'w') as f:
    for i, game in enumerate(analysis_games):
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn_string = game.accept(exporter)
        f.write(pgn_string)
        if i < len(analysis_games) - 1:
            f.write('\n\n')  # Separator for multi-game PGN

print(f"Export complete. Analysis saved to {ANALYSIS_PGN}")