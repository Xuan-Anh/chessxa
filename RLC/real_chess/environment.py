import chess
import numpy as np

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5

class Board(object):

    def __init__(self,opposing_agent,FEN=None):
        """
        Chess Board Environment
        Args:
            FEN: str
                Starting FEN notation, if None then start in the default chess position
        """
        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()


    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:

        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
        if self.board.turn:
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.can_claim_draw():
            self.layer_board[7, :, :] = 1

    def update_layer_board(self,move):
        from_row = move.from_square // 8
        from_col = move.from_col % 8
        to_row = move.to_square // 8
        to_col = move.to_col % 8
        piece_index = np.argmax(np.abs(self.layer_board))
        self._prev_layer_board = self.layer_board.copy()
        self.layer_board[piece_index,to_row, to_col] = self.layer_board[piece_index,from_row, from_col]
        self.layer_board[piece_index,from_row, from_col] = 0

    def pop_layer_board(self):
        self.layer_board = self._prev_layer_board.copy()
        self._prev_layer_board = None


    def step(self,action):
        """
        Run a step
        Args:
            action: tuple of 2 integers
                Move from, Move to

        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: int
                Difference in material value after the move
        """
        piece_balance_before = self.get_material_value()
        self.board.push(action)
        self.update_layer_board()
        if self.board.result() == "*":
            reward = 0
            episode_end = False
        elif self.board.result() == "1-0":
            reward = 1
            episode_end = True
        elif self.board.result() == "0-1":
            reward = -1
            episode_end = True
        elif self.board.is_game_over():
            reward = 0
            episode_end = True
        return episode_end, reward

    def get_random_action(self):
        """
        Sample a random action
        Returns: move
            A legal python chess move.

        """
        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves


    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        self.action_space = np.zeros(shape=(64, 64))
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        for move in moves:
            self.action_space[move[0],move[1]] = 1
        return self.action_space

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * np.sum(self.layer_board[0,:,:])
        rooks = 5 * np.sum(self.layer_board[1,:,:])
        minor = 3 * np.sum(self.layer_board[2:4,:,:])
        queen = 9 * np.sum(self.layer_board[4,:,:])
        return pawns + rooks + minor + queen


    def reset(self):
        """
        Reset the environment
        Returns:

        """
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()
        self.init_action_space()