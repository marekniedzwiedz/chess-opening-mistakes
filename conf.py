# Configuration file for the chess game analyzer script

# Usernames (leave empty to skip the platform)
LICHEss_USERNAME = ""  # Lichess username
CHESS_COM_USERNAME = ""  # chess.com username

# User-Agent for API requests (replace with your details to avoid blocks)
USER_AGENT = "chess-analyzer/1.0 (your.email@example.com)"

# Analysis parameters
NUM_GAMES = 500  # Number of recent games to download initially (if no local file)
MIN_OCCURRENCES = 3  # Minimum occurrences for a position to be considered common
MIN_PLY = 8  # Starting from after move 4 (ply 8: after black's 4th move)
EVAL_THRESHOLD = 50  # Centipawns threshold for a mistake (e.g., 40 = 0.4 pawns)
ANALYSIS_TIME = 5  # Time in seconds for Stockfish analysis
THREADS = 8  # Number of CPU threads for Stockfish (adjust based on your machine)
HASH_MB = 5000  # RAM for Stockfish hash table in MB (adjust based on available RAM)

# Paths (adjust if needed; Stockfish path is required)
STOCKFISH_PATH = ""  # Your Stockfish executable path