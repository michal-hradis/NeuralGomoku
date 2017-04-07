#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys
from player import updateBoard, testWin, invertBoard
from train import printBoard


def parseArgs():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run network on imput from camera.")

    args = parser.parse_args()

    for key, val in vars(args).iteritems():
        print("{}: {}".format(key, val))

    return args


def main():
    # Print the arguments
    args = parseArgs()
    allBoards = []
    allMoves = []
    winBoards = []
    lossBoards = []

    gameCounter = 0
    for game in sys.stdin:
        board = np.ones((2, 19, 19), dtype=np.uint8)
        boards = []
        moves = []
        for move in game.split():
            board = invertBoard(board)
            x, y = [int(x) for x in move.split(',')]
            pos = y * 19 + x
            boards.append(board.copy())
            moves.append(pos)
            board[0, y, x] = 0
        win = testWin(pos, board)
        allBoards.extend(boards)
        allMoves.extend(moves)
        if win:
            boards.append(board)
            boards = boards[1:]
            winStart = (len(moves) + 1) % 2
            lossStart = len(moves) % 2
            winBoards.extend(boards[winStart::2])
            lossBoards.extend(boards[lossStart::2])
        gameCounter += 1
        board = invertBoard(board)
        #printBoard(board)
        print(gameCounter, win, len(allBoards), len(winBoards), len(lossBoards))

    allBoards = np.stack(allBoards)
    allMoves = np.asarray(allMoves)
    winBoards = np.stack(winBoards)
    lossBoards = np.stack(lossBoards)

    np.save('allBoards.npy', allBoards)
    np.save('allMoves.npy', allMoves)
    np.save('winBoards.npy', winBoards)
    np.save('lossBoards.npy', lossBoards)
    

if __name__ == "__main__":
    main()
