import chess
import chess.pgn
import sys


def convert_to_csv(file_path):
    print(file_path)
    pgn = open(file_path)

    csv_name = file_path.split(".")[0] + ".csv"
    with open(csv_name, mode="w") as csv:
        current_game = chess.pgn.read_game(pgn)
        while current_game is not None:
            print(current_game)
            board = current_game.board()
            for move in current_game.mainline_moves():
                board.push(move)
                row = convert_board_to_row(board, 1500, "")
                csv.write(row)


def convert_board_to_row(board, elo, is_black) -> str:
    pass


if __name__ == "__main__":
    print(sys.argv)

    # get file_path
    path = sys.argv[1]
    convert_to_csv(path)
