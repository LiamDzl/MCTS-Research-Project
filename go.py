import torch
import torch.nn.functional as F
import numpy as np
 
border = "🌳"
komi = 7

def show(board):
    print("")
    for column in range(10):
        print(border, end="")

    print(border)

    for row in range(9):
        print(border, end="")
        for column in range(9):
                if board[row, column] == 0:
                    print("🟫", end="")
                elif board[row, column] == 1:
                    print("⚫", end="")
                elif board[row, column] == 2:
                    print("⚪", end="")
        print(border)

    for column in range(10):
        print(border, end="")

    print(border)

    print("")

class Group():
    def __init__(self):
        self.members = set()
        self.liberties = set()
 
class State():
    def __init__(self):
        self.board =  np.zeros((9, 9), dtype=np.int8)
        self.to_move = 1
 
    def show(self, prior, posterior):

        prior = prior.reshape(83)
        posterior = posterior.reshape(83)

        prior_stones = prior[:-2].reshape(9, 9)
        prior_pass = prior[-2]
        prior_value = prior[-1]

        posterior_stones = posterior[:-2].reshape(9, 9)
        posterior_pass = posterior[-2]
        posterior_value = posterior[-1]

        def cell(value):
            if value > 10/81:
                return "🟥"
            elif value > 1/81:
                return "🟦"
            else:
                return "⬛"

        blank = "  "

        print("")
        for column in range(10):
            print(border, end="")
        print(border)

        for row in range(9):
            print(border, end="")
            for column in range(9):
                if self.board[row, column] == 0:
                    print("🟫", end="")
                elif self.board[row, column] == 1:
                    print("⚫", end="")
                elif self.board[row, column] == 2:
                    print("⚪", end="")
            print(border, end="")

            print("    ", end="")
            for column in range(9):
                print(cell(prior_stones[row, column]), end="")
            print("  " + (cell(prior_pass) if row == 4 else blank), end="")

            print("  ", end="")
            for column in range(9):
                print(cell(posterior_stones[row, column]), end="")
            print("  " + (cell(posterior_pass) if row == 4 else blank), end="")

            print("")

        for column in range(10):
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

        print("")
 
    def action(self, move): # move either a pass or a tuple (row, column) for stone
        if move == "pass":
            self.turn()
            return True
 
        resulting_board = self.resolve(move)
        if resulting_board is None:
            print("❌ Invalid Move.")
            return False
 
        self.board = resulting_board
        self.turn()
        return True
 
    def turn(self):
        if self.to_move == 1:
            self.to_move = 2
        else:
            self.to_move = 1
 
    def neighbours(self, row, column):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, column + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                yield nr, nc
 
    def opponent(self, colour):
        return 2 if colour == 1 else 1
 
    def get_group(self, point, board=None):
        if board is None:
            board = self.board
 
        row, column = point
        colour = board[row, column]
        if colour == 0:
            return None
 
        group = Group()
        stack = [point]
 
        while stack:
            r, c = stack.pop()
            if (r, c) in group.members:
                continue
            group.members.add((r, c))
 
            for nr, nc in self.neighbours(r, c):
                neighbour_colour = board[nr, nc]
                if neighbour_colour == 0:
                    group.liberties.add((nr, nc))
                elif neighbour_colour == colour and (nr, nc) not in group.members:
                    stack.append((nr, nc))
 
        return group
 
    def resolve(self, move):
        row, column = move
        if self.board[row, column] != 0:
            return None # occupied
 
        colour = self.to_move
        opponent_colour = self.opponent(colour)
 
        trial_board = self.board.copy()
        trial_board[row, column] = colour
 
        for nr, nc in self.neighbours(row, column):
            if trial_board[nr, nc] == opponent_colour:
                group = self.get_group((nr, nc), trial_board)
                if len(group.liberties) == 0:
                    for mr, mc in group.members:
                        trial_board[mr, mc] = 0
 
        own_group = self.get_group((row, column), trial_board)
        if len(own_group.liberties) == 0:
            return None # suicide
 
        return trial_board
 
class Game():
    def __init__(self):
        self.state = State()
        self.trajectory = [self.state.board.copy()]
        self.moves = []
        self.history = set()
        self.history.add(self.state.board.tobytes())
        self.passes = 0
        self.result = 0
        self.black_score = 0
        self.white_score = 0
 
    def is_legal(self, move):
        if move == "pass":
            return True
 
        resulting_board = self.state.resolve(move)
        if resulting_board is None:
            return False  # occupied or suicide
 
        if resulting_board.tobytes() in self.history:
            return False  # superko: this exact position has occurred before
 
        return True
 
    def legal_moves(self):
        moves = ["pass"]
        for row in range(9):
            for column in range(9):
                if self.is_legal((row, column)):
                    moves.append((row, column))
        return moves

    def compute_mask(self, moves):
        mask = np.zeros((9, 9), dtype=bool)
        for move in moves:
            if move != "pass":
                row, column = move
                mask[row, column] = True

        flat = mask.reshape(81)
        full_mask = np.append(flat, True)
        return torch.from_numpy(full_mask)
 
    def play(self, move):
        if move == "pass":
            self.passes += 1
            self.state.turn()
            self.moves.append(move)
            self.trajectory.append(self.state.board.copy())
            self.history.add(self.state.board.tobytes())
            return True
 
        if not self.is_legal(move):
            print("\n❌ Illegal Move.\n")
            return False
 
        new_board = self.state.resolve(move)
        self.state.board = new_board
        self.state.turn()
 
        self.passes = 0
        self.moves.append(move)
        self.trajectory.append(self.state.board.copy())
        self.history.add(self.state.board.tobytes())
        return True
 
    def is_over(self): # check if game has ended. returns true / false
        if self.passes >= 2:
            self.result = self.determine_winner()

        return self.passes >= 2

    def score(self):
        board = self.state.board
        black_score = 0
        white_score = 0
        visited = set()

        for row in range(9):
            for column in range(9):
                color = board[row, column]
                if color == 1:
                    black_score += 1
                elif color == 2:
                    white_score += 1
                elif (row, column) not in visited:
                    region = set()
                    borders = set()
                    stack = [(row, column)]
                    while stack:
                        r, c = stack.pop()
                        if (r, c) in region:
                            continue
                        region.add((r, c))
                        for nr, nc in self.state.neighbours(r, c):
                            neighbour_colour = board[nr, nc]
                            if neighbour_colour == 0:
                                if (nr, nc) not in region:
                                    stack.append((nr, nc))
                            else:
                                borders.add(neighbour_colour)

                    visited |= region
                    if borders == {1}:
                        black_score += len(region)
                    elif borders == {2}:
                        white_score += len(region)

        return black_score, white_score

    def determine_winner(self):
        black_score, white_score = self.score()
        white_score += komi

        self.black_score = black_score
        self.white_score = white_score

        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return 2
        else:
            return 0
 
    def parse_input(self, raw):
        raw = raw.strip()
        if raw.lower() == "pass":
            return "pass"
        if len(raw) == 2 and raw.isdigit():
            row, column = int(raw[0])-1, int(raw[1])-1
            if 0 <= row < 9 and 0 <= column < 9:
                return (row, column)
        return None
 
    def get_human_move(self):
        player = "Black" if self.state.to_move == 1 else "White"
        while True:
            raw = input(f"{player}: ")
            if raw == "end":
                return "end"
            move = self.parse_input(raw)
            if move is not None:
                return move
            
            print("\n❌ Invalid Input.\n")