import numpy as np
import json

class AlphaBeta:
    """
    Alpha-Beta pruning with heuristic so it chooses the fastest win (a faster win get assigned a better value). This
    version of AlphaBeta works with memory.
    """

    def __init__(self, board, max_depth=4, memory_file=None):
        """"
        origin must be Node class
        position_number_of_moves is used only if a non_empty board is passed to the class
        """
        self.Board = board
        self.max_depth = max_depth
        self.maximum_number_of_moves = board.width * board.height
        if memory_file is None:
            self.memory= {}
        else:
            self.memory = json.load(memory_file)

    # Note: board.undo() before each return insures this look ahead returns the board to its original state
    def __alpha_beta(self, board, node, alpha, beta, depth):
        current_player = board.current_player
        color = self.current_player_color(current_player)
        key = hash(board)

        # Check memory if the node was previously evaluated
        if node.retrieve(key=key) is not None:
            # Check if the previous state was better than current evaluation
            if node.lower_bound >= beta: return node.lower_bound
            if node.upper_bound <= alpha: return node.upper_bound

            # Otherwise, get the shortest window with current and previous evaluation
            alpha = max(alpha, node.lower_bound)
            beta = min(beta, node.upper_bound)

        if depth == 0 or node.is_terminal_node():
            return self.heuristic_value(
                board=board,
                node=node,
                alpha=alpha,
                beta=beta,
                color=color
            )

        # A look ahead, depth-first search
        value = -self.maximum_number_of_moves
        for move in node.moves_leading_to_child():
            board.try_place_piece(move)
            depth -= 1
            child = Node(board=board, lower_bound=-alpha, upper_bound=-beta, memory=self.memory)
            value = max(value, -self.__alpha_beta(board, child, -alpha, -beta, depth))
            depth = self.max_depth
            alpha = max(alpha, value)
            if alpha >= beta: # Skip childs nodes which are certain to lead to a worse position
                break
        board.undo()
        return value


    def calculate_move(self): # Note: only works if legal moves is not empty. Check if game is over before calling this
        origin = self.create_node()
        key = hash(self.Board)
        # Check memory if the node was previously evaluated
        bounds = origin.retrieve(key)
        if bounds is not None:
            stored_upper_bound = bounds[0]
            stored_lower_bound = bounds[1]
            if origin.lower_bound <= stored_lower_bound:
                origin.lower_bound = stored_lower_bound
            if origin.upper_bound >= stored_upper_bound:
                origin.upper_bound = stored_upper_bound

        depth = self.max_depth
        look_ahead_values = [0] * self.Board.width
        for legal_move in origin.moves_leading_to_child():
            self.Board.act(legal_move, current_sign=self.Board.current_player)
            look_ahead_values[legal_move] = self.__alpha_beta(
                board=self.Board,
                node=origin,
                alpha=origin.lower_bound,
                beta=origin.upper_bound,
                depth=depth
            )
            self.Board.undo_action()
        if self.is_minimizing_player(current_player=self.Board.current_player):
            return np.argmin(look_ahead_values)
        elif self.is_maximizing_player(current_player=self.Board.current_player):
            return np.argmax(look_ahead_values)

    def create_node(self, lower_bound=None, upper_bound=None):
        return Node(board=self.Board, memory=self.memory, lower_bound=lower_bound, upper_bound=upper_bound)

    @staticmethod
    def opponent_sign(current_player):
        if current_player == "X":
            return "O"
        else:
            return "X"

    @staticmethod
    def current_player_color(current_player):
        if current_player == "X":
            return 1
        else:
            return -1

    #TODO make sure alpha and beta implement properly
    #TODO implement transposition table
    def heuristic_value(self, board, node, alpha, beta, color):
        if node.is_terminal_node():
            board.undo()
            return color * (self.maximum_number_of_moves - board.number_of_moves_played())/2
        else: # Check if next move is winning
            for move in range(board.width):
                move_allowed, row = board.try_place_piece(move)

                if move_allowed and board.is_game_over()[0]:
                    value = -color * (self.maximum_number_of_moves - board.number_of_moves_played()) / 2
                    board.undo()
                    return value
        """
        If next move doesn't end the game, then the value is a lower or upper bound estimate of the true value, 
        depending if the player is maximizing or minimizing. Maximizing player can expect at worst the lower_bound
        of the window, which is alpha. Minimizing player can expect at worst the upper bound of the window, which is beta.
        """
        if self.is_maximizing_player(board.current_player):
            board.undo()
            return alpha
        elif self.is_minimizing_player(board.current_player):
            board.undo()
            return beta

    @staticmethod
    def is_maximizing_player(current_player):
        if current_player == "X":
            return True
        else:
            return False

    @staticmethod
    def is_minimizing_player(current_player):
        if current_player == "O":
            return True
        else:
            return False


class Node:
    """"
    Node of the game tree.
    """
    def __init__(self, board, memory, lower_bound, upper_bound):
        self.memory = memory
        self.current_player = board.current_player
        max_bound = board.width * board.height
        if lower_bound is None:
            self.lower_bound = -max_bound
        else:
            self.lower_bound = lower_bound
        if upper_bound is None:
            self.upper_bound = max_bound
        else:
            self.upper_bound = upper_bound
        self.Board = board

    # Value is a list of 3 elements to have shortest size possible
    # 0 is best move from that node
    # 1 is value of that node
    # 2 is flag: True = exact value, False = estimate value (lower or upper bound)
    # both can have same value, in which case the estimate for the node is the true value.
    def retrieve(self, key):
        for stored_key, value in self.memory.items():
            if stored_key == key:
                if self.current_player is AlphaBeta.is_maximizing_player(self.current_player):
                    # return lower bound
                    return value[0]
                elif self.current_player is AlphaBeta.is_minimizing_player(self.current_player):
                    # return upper bound
                    return value[1]
                else:
                    print("player must be either 'X' or 'O'")
        else:
            return None

    def store(self, move:np.int8, value:np.int8, flag:bool):
        """"
        We require that the tuple store takes minimal size.
        """
        key = hash(self.Board)
        self.memory[key] = (move, value, flag)
        return self.memory

    def moves_leading_to_child(self):
        legal_moves = []
        for i in range(self.Board.height):
            row = self.Board.height - i - 1  # Pieces are added from bottom to up
            for column in range(self.Board.width):
                if self.Board.board[row][column] == ' ':
                    legal_moves.append(column + 1) # move = column + 1, a nu
        return legal_moves

    def is_terminal_node(self):
        if self.Board.is_game_over()[0]:
            return True
        else:
            return False