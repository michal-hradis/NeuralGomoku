#!/usr/bin/env python
from __future__ import print_function

import caffe
import numpy as np
import sys
from player import simpleSearchPlayer, updateBoard, testWin, invertBoard
from time import time
from termcolor import colored

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run network on imput from camera.")

    args = parser.parse_args()

    for key, val in vars(args).iteritems():
        print("{}: {}".format(key, val))

    return args


def printBoard(board):
    board = board.transpose(1, 2, 0)
    for line in board:
        for pos in line:
            if pos[0] == -1:
                sys.stdout.write(colored('X', 'red'))
            elif pos[0] == 0:
                sys.stdout.write(colored('X', 'green'))
            elif pos[1] == -1:
                sys.stdout.write(colored('O', 'red'))
            elif pos[1] == 0:
                sys.stdout.write(colored('O', 'yellow'))
            else:
                print(' ', end='')
        print('')


def game(player1, player2, score):

    board = np.ones((2, 19, 19), dtype=np.float32)
    for i in range(1000):
        t1 = time()
        scores = player1.scoreMoves(board) ** 12
        t2 = time()
        scores /= scores.sum()
        pos = np.random.choice(scores.size, p=scores)
        updateBoard(pos, board)
        print(t2-t1, 'SCORE:', score)

        temp = board[:, pos / 19, pos % 19].copy()
        board[0, pos / 19, pos % 19] = -1
        raw_input('Enter')
        sys.stdout.write('\x1b[2J\x1b[H')
        if i % 2 == 0:
            printBoard(board)
        else:
            printBoard(invertBoard(board))
        board[:, pos / 19, pos % 19] = temp

        if testWin(pos, board):
            print("WINNER WINNER WINNER WINNER WINNER WINNER WINNER WINNER ")
            return i % 2

        board = invertBoard(board)
        player1, player2 = player2, player1

    return -1

def main():
    import cProfile

    # Print the arguments
    args = parseArgs()

    caffe.set_device(0)
    caffe.set_mode_gpu()

    pNet = caffe.Net('../nets/pNet.prototxt', '../nets/pNet.300000.model', caffe.TEST)
    qNet = caffe.Net('../nets/qNet.prototxt', '../nets/qNet.50000.model', caffe.TEST)
    player1 = simpleSearchPlayer(pNet=pNet, qNet=qNet, pathCount=256, depth=3)

    pNet = caffe.Net('../nets/pNet.prototxt', '../nets/pNet.300000.model', caffe.TEST)
    qNet = caffe.Net('../nets/qNet.prototxt', '../nets/qNet.50000.model', caffe.TEST)
    player2 = simpleSearchPlayer(pNet=pNet, qNet=qNet, pathCount=256, depth=3)

    score = [0, 0]
    for gameID in range(1000):
        winner = game(player1, player2, score)
        score[winner] += 1



if __name__ == "__main__":
    main()
