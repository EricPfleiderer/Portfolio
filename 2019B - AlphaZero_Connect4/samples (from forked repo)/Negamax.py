#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This heuristic algorithm is based on negamax, which is a variant of a minimax
algorithm. Negamax can be used due to zero-sum property of Connect-4. Heuristic
algorithm is needed, because Connect Four has around 4*10^12 (4 trillion)
different possible games.
"""
import json
import numpy as np

class Negamax:

    '''
    board, 2D Matrix: game layout

    max_depth, Integer: maximum depth visited during Negamax algorithm. Higher values increase probability of selecting
    the correct move. However, the time required by the algorithm evolves exponentially with this parameter.
    '''
    def __init__(self, board, max_depth=4, memory_file=None):
        self.__listed_indexes = board.segment_indexes
        self.__weights = [1, 8, 128, 99999]
        self.__max_depth = max_depth
        #TODO create a memory file
        #TODO implement alpha-beta pruning algorithm
        #TODO change memory so it appends to a json file
        if memory_file is None:
            self.__evaluated = {}
        else:
            self.__evaluated = json.load(memory_file)
        self.best_score = board.width * board.height


    '''
    Method returns the best move to play (column number from 1-7) and the best score (Integer).
    The board parameter MUST be a Board object. Board objects have a board attribute that represent the gamestate.
    The gamestate is represented as a list in the AlphaGo Zero architecture, while it is represented as a 6x7 matrix 
    in the Negamax architecture. 
    '''
    def __negamax(self, board, curr_sign, opponent_sign, depth=0):

        #The Negamax algorithm operates on a Board type object. If the board passed as parameter is in list ,
        #it must first be converted.
        if isinstance(board, list):
            board = self.convertListToMatrix(board)

        hashed_board = hash(board)

        if hashed_board in self.__evaluated:
            return None, self.__evaluated[hashed_board]

        #Reached maximum allowed depth.
        if depth == self.__max_depth:
            score = self.__evaluate(board.board, curr_sign, opponent_sign)
            self.__evaluated[hashed_board] = score
            return None, score

        best_score = float('-inf')
        best_move = None

        #TODO make the board size dynamic
        for x in range(1, 8):
            move = x
            move_allowed, row = board.try_place_piece(move, curr_sign)

            if not move_allowed:
                continue

            game_over, winner = board.is_game_over(board.board, curr_sign, opponent_sign, (move-1, row))
            if game_over:
                if winner == curr_sign:
                    best_subscore = 9999999999
                elif winner == opponent_sign:
                    best_subscore = -9999999999
                else:
                    best_subscore = 0
            else:
                best_submove, best_subscore = self.__negamax(board, opponent_sign, curr_sign, depth + 1)
                best_subscore *= -1
            board.undo_one_move()

            if best_subscore > best_score:
                best_score = best_subscore
                best_move = move

        # Happens when max_depth exceeds number of possible moves
        if best_move is None:
            best_score = self.__evaluate(board.board, curr_sign, opponent_sign)

        self.__evaluated[hashed_board] = best_score

        return best_move, best_score

    def __evaluate(self, board, curr_sign, opponent_sign):
        """Counts and weighs longest connected checker chains
        which can lead to win"""

        curr_score = 0
        opp_score = 0

        for indexes in self.__listed_indexes:
            # indexes contains four board indexes as tuples

            curr_count = 0
            opp_count = 0

            for index in indexes:
                v = board[index[0]][index[1]]
                if v == curr_sign:
                    curr_count += 1
                elif v == opponent_sign:
                    opp_count += 1

            if curr_count > 0 and opp_count > 0:
                continue
            elif curr_count > 0:
                curr_score += curr_count * self.__weights[curr_count - 1]
            elif opp_count > 0:
                opp_score += opp_count * self.__weights[opp_count - 1]

        difference = curr_score - opp_score

        return difference

    def calculate_move(self, board, curr_sign, opponent_sign):
        move, score = self.__negamax(board, curr_sign, opponent_sign)
        #print(score)
        return move

    '''
    Interface that binds AlphaGo Zero architecture with NegaMax architecture.
    Takes as input a 42 entry list containing 1,0 and -1 and converts it to a 6x7 matrix containing X and Os.
    '''
    def convertListToMatrix(self, board, width = 7, height = 6):

        new_board = [[' ' for x in range(width)] for y in range(height)]

        if len(board) == width*height:
            counter = 0
            for x in range(width):
                for y in range(height):
                    if board[counter] is 1:
                        new_board[y][x] = "X"
                    elif board[counter] is -1:
                        new_board[y][x] = "O"
                    counter+=1

            return [[' ' for x in range(width)] for y in range(height)]

        else:
            raise ValueError("Length of list cannot be converted to matrix specified width and height (7 and 6 by default).")


# TODO:Prefer games that win faster or lose later
