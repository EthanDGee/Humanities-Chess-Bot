import chess
import time
from src.play.ui.ui import Ui
from src.play.player.human_player import HumanPlayer

class Cli(Ui):
    """Command-line interface for chess games with its own game loop."""

    LOOP_INTERVAL = 0.1  # seconds between loop iterations

    def __init__(self, game):
        self.game = game
        self.move_history = []

    def run(self):
        """Start the CLI game loop, similar to GUI's after loop."""
        self._game_loop()

    def reset_ui(self):
        self.move_history = []
        print("[UI RESET]\n")

    def display_board(self):
        print(self.game.board, "\n")

    def show_message(self, message):
        print(f"[MESSAGE] {message}")

    def update_scores(self):
        white_score, black_score = self.game.get_scores()
        print(f"[SCORES] White: {white_score}  Black: {black_score}\n")

    def update_move_list(self, move_san):
        self.move_history.append(move_san)
        print(f"[MOVE] {move_san}\n")

    def prompt_move(self, player):
        """Prompt human player for a legal move."""
        legal_moves = list(self.game.board.legal_moves)
        while True:
            move_str = input(f"{player.name}'s move (UCI, e.g., e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_str)
                if move in legal_moves:
                    return move
                print("Illegal move! Try again.")
            except Exception as e:
                print(f"Invalid input: {e}")

    def _game_loop(self):
        self.reset_ui()
        self.display_board()

        while not self.game.is_over():
            current_player = self.game.current_player

            # Handle human vs bot
            if isinstance(current_player, HumanPlayer):
                move = self.prompt_move(current_player)
            else:
                move = current_player.get_move(self.game.board)

            if move:
                move_san = self.game.apply_move(move)
                self.update_move_list(move_san)
                self.display_board()
                self.update_scores()

            # Unified timer logic
            timeout_winner = self.game.update_timer()
            print(f"[TIMERS] White: {self.game.white_time_left:.1f}s  Black: {self.game.black_time_left:.1f}s")
            if timeout_winner:
                print(f"Game Over: {timeout_winner.name} wins on time!")
                break

            time.sleep(self.LOOP_INTERVAL)

        # Game over
        result = self.game.result()
        if result == "1-0":
            print(f"Game Over! {self.game.white_player.name} wins!")
        elif result == "0-1":
            print(f"Game Over! {self.game.black_player.name} wins!")
        else:
            print("Game Over! Draw!")
