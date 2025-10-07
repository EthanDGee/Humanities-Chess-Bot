from abc import ABC, abstractmethod

class Ui(ABC):
    """Abstract UI interface that any user interface (CLI, GUI, TUI) should implement."""

    @abstractmethod
    def run(self):
        """Start the game loop."""
        pass

    @abstractmethod
    def display_board(self, game):
        """Render the current board state."""
        pass

    @abstractmethod
    def show_message(self, message):
        """Show a message to the user (game over, invalid move, etc.)."""
        pass

    @abstractmethod
    def update_scores(self, white_score, black_score):
        """Update player scores."""
        pass

    @abstractmethod
    def update_move_list(self, move_san):
        """Update the move history display."""
        pass

    @abstractmethod
    def reset_ui(self):
        """Reset UI elements like move history or highlights."""
        pass
