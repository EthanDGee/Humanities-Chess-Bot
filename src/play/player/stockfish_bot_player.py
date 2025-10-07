import chess
import chess.engine
import shutil

from src.play.player.player import Player

# ================================================================
# StockfishPlayer
# A chess bot that uses the Stockfish engine.
# ================================================================

# ----------------------------
# Setup Instructions:
# ----------------------------
# 1. Install Stockfish:
#    - On Linux (Debian/Ubuntu):
#        sudo apt install stockfish
#    - On macOS using Homebrew:
#        brew install stockfish
#    - On Windows:
#        Download from https://stockfishchess.org/download/ and place the executable in a folder added to PATH.
#
# 2. Install Python dependencies:
#    pip install chess
#
# 3. Integrate with your game framework:
#    - Import StockfishPlayer and pass it to your game loop.
#    - You can adjust `skill_level` (0–20) for difficulty.
#    - You can adjust `time_limit` to control thinking time per move.

class StockfishPlayer(Player):
    def __init__(self, name="Stockfish", color=True, skill_level=10, time_limit=0.5):
        """
        color: True for white, False for black
        skill_level: 0–20 (Stockfish difficulty)
        time_limit: seconds per move (float)
        """
        super().__init__(name, color)
        self.pending_move = None
        self.time_limit = time_limit
        self.skill_level = skill_level

        # Locate Stockfish binary
        self.engine_path = shutil.which("stockfish")
        if self.engine_path is None:
            raise FileNotFoundError(
                "Stockfish binary not found in PATH. Try installing it with `sudo apt install stockfish`."
            )

        # Initialize Stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.engine.configure({"Skill Level": skill_level})

    def get_move(self, board: chess.Board):
        # Only make a move if it's this player's turn
        if board.turn != self.color:
            return None

        try:
            result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
            return result.move
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None

    def close(self):
        # Safely terminate engine
        try:
            self.engine.quit()
        except Exception:
            pass  # Engine may already be closed

    def __del__(self):
        self.close()
