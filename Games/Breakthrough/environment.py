import numpy as np
import copy

border = "🌳"

def show(board):
    print("")
    for column in range(9):
        print(border, end="")
    print(border)

    for row in range(8):
        print(border, end="")
        for column in range(8):
            if board[row, column] == 0:
                print("🟫", end="")
            elif board[row, column] == 1:
                print("⚪", end="")
            elif board[row, column] == 2:
                print("⚫", end="")
        print(border)

    for column in range(9):
        print(border, end="")
    print(border)

    print("")

class State():
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.float32)
        self.board[0:2, :] = 2
        self.board[6:8, :] = 1
        self.to_move = 1
        self.mask = np.zeros(192, dtype=np.float32)

    def show(self, prior, posterior):
        prior = prior.reshape(65)
        posterior = posterior.reshape(65)

        prior_squares = prior[:-1].reshape(8, 8)
        prior_value = prior[-1]

        posterior_squares = posterior[:-1].reshape(8, 8)
        posterior_value = posterior[-1]

        def cell(value):
            if value > 10/64:
                return "🟥"
            elif value > 1/64:
                return "🟦"
            else:
                return "⬛"

        print("")
        for column in range(9):
            print(border, end="")
        print(border)

        for row in range(8):
            print(border, end="")
            for column in range(8):
                if self.board[row, column] == 0:
                    print("🟫", end="")
                elif self.board[row, column] == 1:
                    print("⚪", end="")
                elif self.board[row, column] == 2:
                    print("⚫", end="")
            print(border, end="")

            print("    ", end="")
            for column in range(8):
                print(cell(prior_squares[row, column]), end="")

            print("    ", end="")
            for column in range(8):
                print(cell(posterior_squares[row, column]), end="")

            print("")

        for column in range(9):
            print(border, end="")
        print(border)

        print("")

        if prior_value >= 0: 
            print(f"Network: +{prior_value:.3f}")
        else:
            print(f"Network: {prior_value:.3f}")

        if posterior_value >= 0: 
            print(f"Search: +{posterior_value:.3f}")
        else:
            print(f"Search: {posterior_value:.3f}")

    def turn(self):
        if self.to_move == 1:
            self.to_move = 2
        else:
            self.to_move = 1

    def action(self, move):
        resulting_board = self.resolve(move)
        if resulting_board is None:
            print("❌ Invalid Move.")
            return False

        self.board = resulting_board
        self.turn()
        return True

    def opponent(self, colour):
        return 2 if colour == 1 else 1

    def resolve(self, move):
        row, column, direction = move
        colour = self.to_move

        if self.board[row, column] != colour:
            return None

        nr = row - 1 if colour == 1 else row + 1
        nc = column + (direction - 2)
        if not (0 <= nr < 8 and 0 <= nc < 8):
            return None

        target = self.board[nr, nc]
        if direction == 2:
            if target != 0:
                return None
        elif target == colour:
            return None

        trial_board = self.board.copy()
        trial_board[row, column] = 0
        trial_board[nr, nc] = colour
        return trial_board


class Game():
    def __init__(self):
        self.state = State()
        self.moves = []
        self.result = 0
        self.state.mask = self.compute_mask(self.legal_moves())
        self.trajectory = [copy.deepcopy(self.state)]

    def play(self, move):
        if not self.is_legal(move):
            print("\n❌ Illegal Move.\n")
            return False

        self.state.board = self.state.resolve(move)
        self.state.turn()

        self.moves.append(move)
        self.state.mask = self.compute_mask(self.legal_moves())
        self.trajectory.append(copy.deepcopy(self.state))

        return True

    def is_legal(self, move):
        return self.state.resolve(move) is not None

    def legal_moves(self):
        moves = []
        colour = self.state.to_move

        for row in range(8):
            for column in range(8):
                if self.state.board[row, column] == colour:
                    for direction in (1, 2, 3):
                        if self.is_legal((row, column, direction)):
                            moves.append((row, column, direction))

        return moves

    def compute_mask(self, moves):
        mask = np.full(192, -np.inf, dtype=np.float32)
        for move in moves:
            mask[self.move_to_index(move)] = 0.0
        return mask

    def is_over(self):
        self.result = self.determine_winner()
        return self.result != 0

    def determine_winner(self):
        board = self.state.board

        if np.any(board[0, :] == 1) or not np.any(board == 2):
            return 1
        if np.any(board[7, :] == 2) or not np.any(board == 1):
            return 2
        if np.all(np.isinf(self.state.mask)):
            return self.state.opponent(self.state.to_move)

        return 0

    def parse_input(self, raw):
        raw = raw.strip().replace(" ", "").replace(",", "")
        if len(raw) == 3 and raw.isdigit():
            row, column, direction = int(raw[0]) - 1, int(raw[1]) - 1, int(raw[2])
            if 0 <= row < 8 and 0 <= column < 8 and direction in (1, 2, 3):
                return (row, column, direction)
        return None

    def get_human_move(self):
        player = "White" if self.state.to_move == 1 else "Black"
        while True:
            raw = input(f"{player}: ")
            if raw == "end":
                return "end"
            move = self.parse_input(raw)
            if move is not None:
                return move

            print("\n❌ Invalid Input.\n")

    def move_to_index(self, move):
        row, column, direction = move
        return 3 * (8 * row + column) + (direction - 1)

    def index_to_move(self, index):
        square, direction = divmod(index, 3)
        row, column = divmod(square, 8)
        return (row, column, direction + 1)