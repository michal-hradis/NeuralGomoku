#!/usr/bin/env python
from __future__ import print_function

import caffe
import numpy as np
from numba import jit


def invertBoard(board):
    return board[::-1, :, :]

@jit
def choice(p):
    p = np.cumsum(p)
    p /= p[-1]
    th = np.random.rand()
    return np.argmax(p > th)

@jit
def testWin(move, board):
    board = board[0]
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    moveY = move / board.shape[1]
    moveX = move % board.shape[1]
    for dx, dy in directions:
        counter = 0
        for i in range(-4, 5):
            x = moveX + i * dx
            y = moveY + i * dy
            if x >= 0 and y >= 0 and x < 19 and y < 19:
                if board[y, x] == 0:
                    counter += 1
                else:
                    counter = 0
                if counter == 5:
                    return True
    return False

@jit
def testWins(moves, boards):
    wins = np.zeros(len(moves), dtype=np.bool)
    for i in range(len(moves)):
        wins[i] = testWin(moves[i], boards[i])
    return wins

@jit
def updateBoards(moves, boards, invBoards):
    boards = boards.reshape(boards.shape[0], 2, -1)
    invBoards = invBoards.reshape(boards.shape[0], 2, -1)
    for i in range(len(moves)):
        move = moves[i]
        if np.count_nonzero(boards[i, :, move]) == 2:
            boards[i, 0, move] = 0
            invBoards[i, 1, move] = 0

@jit
def updateBoard(move, board):
    board = board.reshape(2, -1)
    if np.count_nonzero(board[:, move]) == 2:
        board[0, move] = 0


class simpleSearchPlayer(object):

    def __init__(self, pNet, qNet, pathCount=512, depth=10):
        self.pNet = pNet
        self.qNet = qNet
        self.depth = depth
        self.pathCount = pathCount
        self.boardSize = 19
        self.outLayer = 'out'
        self.inputLayer = 'data'

        self.qNet.blobs[self.inputLayer].reshape(
            self.pathCount, 2, self.boardSize, self.boardSize)
        self.qNet.reshape()

        self.pNet.blobs[self.inputLayer].reshape(
            self.pathCount, 2, self.boardSize, self.boardSize)
        self.pNet.reshape()

    #@jit
    def generateMoves(self, boards):
        self.pNet.blobs[self.inputLayer].data[...] = boards
        self.pNet.forward()
        probs = self.pNet.blobs[self.outLayer].data
        probs = probs.reshape(probs.shape[0], -1)

        moves = []
        for p in probs:
            pos = choice(p)
            moves.append(pos)

        return np.asarray(moves)

    def evaluateBoards(self, boards):
        self.qNet.blobs[self.inputLayer].data[...] = boards
        self.qNet.forward()
        return self.qNet.blobs[self.outLayer].data.reshape(-1)

    def scoreMoves(self, board):
        pleyerBoards = np.tile(board, [self.pathCount, 1, 1, 1])
        opponentBoards = np.tile(invertBoard(board), [self.pathCount, 1, 1, 1])
        playerWon = np.asarray([False] * self.pathCount)
        opponentWon = np.asarray([False] * self.pathCount)

        swapped = False
        for i in range(self.depth):
            moves = self.generateMoves(pleyerBoards)
            updateBoards(moves, pleyerBoards, opponentBoards)
            wins = testWins(moves, pleyerBoards)
            playerWon = np.logical_or(
                playerWon,
                np.logical_and(wins, np.logical_not(opponentWon)))

            pleyerBoards, opponentBoards = opponentBoards, pleyerBoards
            playerWon, opponentWon = opponentWon, playerWon

            swapped = not swapped
            if i == 0:
                startMoves = moves.copy()

        pathScores = self.evaluateBoards(pleyerBoards)
        moveScores = np.zeros(self.boardSize * self.boardSize)
        if swapped:
            pathScores = 1.0 - pathScores
            playerWon, opponentWon = opponentWon, playerWon
        #rint(playerWon.sum(), opponentWon.sum())
        #pathScores[...] = 0.01
        pathScores[playerWon] = 1.0
        pathScores[opponentWon] = 0.0
        for pos, score in zip(startMoves, pathScores ):
            moveScores[pos] += score
        return moveScores


