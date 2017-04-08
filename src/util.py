from __future__ import print_function
import sys
from termcolor import colored

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
