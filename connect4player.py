"""
This Connect Four player uses the Minimax algorithm to choose its next move.
The difficulty level is the number of plies ahead the bot looks.
"""
__author__ = "Branson Jones"  # replace my name with yours
__license__ = "MIT"
__date__ = "February 22, 2024"

import numpy as np


class ComputerPlayer:
    def __init__(self, id, difficulty_level):
        """
        Constructor, takes a difficulty level (the # of plies to look
        ahead), and a player ID that's either 1 or 2 that tells the player what
        its number is.
        """
        self.ID = id  # which number this player is
        self.MAX_PLIES = difficulty_level  # max number of plies

    def pick_move(self, rack):
        """
        Pick the move to make. It will be passed a rack with the current board
        layout, column-major. A 0 indicates no token is there, and 1 or 2
        indicate discs from the two players. Column 0 is on the left, and row 0
        is on the bottom. It must return an int indicating which column to
        drop a disc into.
        """
        shape = (len(rack), len(rack[0]))  # (num_cols, num_rows)
        # Convert to numpy array
        numpy_rack = np.full(shape, rack, dtype=int, order='F')

        if self._rack_full(numpy_rack, shape):
            return None  # no possible move

        i = 0
        best_move = 0
        high_score = -10000000000

        while i < shape[0]:
            # if column is full, skip column
            if self._make_move(numpy_rack, i, shape, self.ID):
                score = self._minimax(numpy_rack, shape, 1)
                if score > high_score:
                    high_score = score
                    best_move = i
                self._unmake_move(numpy_rack, i, shape)
            i += 1

        return best_move

    def _minimax(self, rack, shape, ply):
        """
        A recursive helper method for pick_move() that uses the Minimax algorithm to
        calculate the best possible move.
        :param rack: 2D array of ints representing the current board layout.
        :param shape: (num_cols, num_rows) tuple holding dimensions of rack.
        :param ply: int level of recursion @PRE: Positive, non-zero
        :return: score that this path will yield assuming optimal play
        """
        # check for win
        score = self._board_state_score(rack, shape)
        if score[1]:
            # won on the last move
            return score[0] / ply
        # not a win

        # check for tie
        if self._rack_full(rack, shape):
            # tie game
            return 0

        # check if leaf
        if ply == self.MAX_PLIES:
            # stop recursion
            score = self._board_state_score(rack, shape)
            if score[1]:
                # winner winner
                return score[0]/ply
            return score[0]
        # not tie or leaf

        # who makes the next decision?
        if (ply % 2) == 1:
            # min's turn:
            if self.ID == 1:
                player = 2
            else:
                player = 1
        else:
            # max's turn
            player = self.ID

        i = 0
        high_score = -10000000000
        low_score = 10000000000

        # iterate through columns
        while i < shape[0]:
            # if column is full, skip column
            if self._make_move(rack, i, shape, player):
                score = self._minimax(rack, shape, ply+1)

                if score > high_score:
                    high_score = score
                if score < low_score:
                    low_score = score

                self._unmake_move(rack, i, shape)
            i += 1

        # choose score to return
        if self.ID != player:
            # min's turn: choose the lowest score
            return low_score
        # else, max's turn: choose the highest score
        return high_score

    def _board_state_score(self, rack, shape):
        """
        Determines aggregate score of a board state. Also determines if a player has won in this state.
        :param rack: 2D numpy array rack of state being evaluated
        :param shape: (num_cols, num_rows) tuple holding dimensions of rack.
        :return: (int board_state_score, boolean was_a_win) tuple
        """
        score = 0

        # vertical quartets
        if shape[1] > 3:
            for i in range(shape[0]):
                for j in range((shape[1]-3)):
                    tmp = self._quartet_score(rack[i][j], rack[i][j+1], rack[i][j+2], rack[i][j+3])
                    if tmp[1]:
                        # there is a winner: end score calculation
                        return tmp
                    score += tmp[0]

        # horizontal quartets
        if shape[0] > 3:
            for i in range((shape[0]-3)):
                for j in range(shape[1]):
                    tmp = self._quartet_score(rack[i][j], rack[i+1][j], rack[i+2][j], rack[i+3][j])
                    if tmp[1]:
                        # there is a winner: end score calculation
                        return tmp
                    score += tmp[0]

        # diagonal quartets
        if shape[0] > 3 and shape[1] > 3:
            # bottom-left to top-right
            for i in range((shape[0]-3)):
                for j in range((shape[1]-3)):
                    tmp = self._quartet_score(rack[i][j], rack[i+1][j+1], rack[i+2][j+2], rack[i+3][j+3])
                    if tmp[1]:
                        # there is a winner: end score calculation
                        return tmp
                    score += tmp[0]
            # top-left to bottom-right
            for i in range(3, (shape[0])):
                for j in range((shape[1]-3)):
                    tmp = self._quartet_score(rack[i][j], rack[i-1][j+1], rack[i-2][j+2], rack[i-3][j+3])
                    if tmp[1]:
                        # there is a winner: end score calculation
                        return tmp
                    score += tmp[0]

        return score, False

    def _quartet_score(self, disc1, disc2, disc3, disc4):
        """
        Determines the score of a set of four sequential disks (a quartet).
        Also determines if this quartet represents a win.
        :param disc1: int 0, 1, or 2
        :param disc2: int 0, 1, or 2
        :param disc3: int 0, 1, or 2
        :param disc4: int 0, 1, or 2
        :return: (int quartet_score, boolean was_a_win)
        """
        if disc1 == disc2 == disc3 == disc4:
            # win or empty
            if disc1 == 0:
                # empty
                return 0, False
            if disc1 == self.ID:
                # win for max
                return 1000000000, True
            else:
                # win for min
                return -1000000000, True

        # check for both 1 and 2 in same quartet
        if ((disc1 == 1 or disc2 == 1 or disc3 == 1 or disc4 == 1)
                and (disc1 == 2 or disc2 == 2 or disc3 == 2 or disc4 == 2)):
            return 0, False

        # determine who non-zero discs belong to
        if disc1 == 1 or disc2 == 1 or disc3 == 1 or disc4 == 1:
            pid = 1
        else:
            pid = 2

        # determine number of zeroes (should be 1-3)
        zeros = 0
        if disc1 == 0:
            zeros += 1
        if disc2 == 0:
            zeros += 1
        if disc3 == 0:
            zeros += 1
        if disc4 == 0:
            zeros += 1

        if zeros == 3:
            # just one tile
            if pid == self.ID:
                return 1, False
            else:
                return -1, False
        if zeros == 2:
            # two tiles
            if pid == self.ID:
                return 10, False
            else:
                return -10, False
        # zeros == 1: three tiles
        if pid == self.ID:
            return 100, False
        else:
            return -100, False

    @staticmethod
    def _make_move(rack, col, shape, player):
        """
        Makes move by altering rack. Will not make move if column is full.
        :param rack: 2D numpy array rack
        :param col: 0-indexed column number to make move at.
            @PRE: must be valid column number.
        :param shape: (num_cols, num_rows) tuple holding dimensions of rack.
        :param player: 1 or 2
        :return: True if move was made, False if column is full
        """
        if rack[col][shape[1]-1] != 0:
            return False
        i = 0
        while i < shape[1]:
            if rack[col][i] == 0:
                rack[col][i] = player
                return True
            i += 1

    @staticmethod
    def _unmake_move(rack, col, shape):
        """
        Unmakes a move by altering rack. Will not unmake move if col specifies an empty column.
        :param rack: 2D numpy array rack
        :param col: 0-indexed column number to make move at.
            @PRE: must be valid column number.
        :param shape: (num_cols, num_rows) tuple holding dimensions of rack.
        :return: True if move was unmade, False if col is empty
        """
        i = shape[1]-1
        while i >= 0:
            if rack[col][i] != 0:
                rack[col][i] = 0
                return True
            i -= 1
        return False

    @staticmethod
    def _rack_full(rack, shape):
        """
        Determines if the rack is full.
        :param rack: 2D numpy array rack
        :param shape: (num_cols, num_rows) tuple holding dimensions of rack.
        :return: True if full, False otherwise
        """
        for i in range(shape[0]):
            if rack[i][shape[1]-1] == 0:
                return False
        return True
