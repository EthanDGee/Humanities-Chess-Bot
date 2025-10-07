import random
from src.play.player.lc0_bot_player import Lc0BotPlayer
from src.play.player.stockfish_bot_player import StockfishPlayer
from src.play.player.human_player import HumanPlayer

from src.play.game.game import Game
from src.play.ui.cli import Cli
from src.play.ui.gui import Gui


def main():
    # Randomly assign human/bots or choose fixed configuration
    if random.choice([True, False]):
        white = Lc0BotPlayer("Leela", color=True, time_limit=1.0)
        black = StockfishPlayer("Stockfish", color=False, skill_level=10, time_limit=0.5)
    else:
        white = StockfishPlayer("Stockfish", color=True, skill_level=10, time_limit=0.5)
        black = Lc0BotPlayer("Leela", color=False, time_limit=1.0)

    # Create the game object
    save_dir = "/home/nate/projects/cs6640_project/data/pgn"
    game = Game(white, black, save_dir=save_dir)

    ui = Gui(game)
    game.ui = ui
    ui.run()

if __name__ == "__main__":
    main()
