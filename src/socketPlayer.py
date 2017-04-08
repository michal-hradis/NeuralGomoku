#!/usr/bin/env python
from __future__ import print_function

import caffe
import numpy as np
import socket
import sys

from player import simpleSearchPlayer
from train import printBoard

"""
This script is used for communicating with game client using sockets
It recieved coordinates, updates board state and sends next move to
AI run by game client.
"""


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ip_addr', default="192.168.56.1",
                        required=True)
    parser.add_argument('-p', '--port', default=27015, type=int,
                        required=True)
    args = parser.parse_args()

    for key, val in vars(args).iteritems():
        print("{}: {}".format(key, val))

    return parser.parse_args()


def disconnect(sock, connection):
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()


def get_response(data):
    print(type(data[0]), data)
    sys.stdout.flush()
    index = 0
    ret = []

    while index < len(data):
        #
        if ord(data[index]) == 255:
            index += 1
        else:
            ret.append(bytearray([]))

            while index < len(data):
                print(ord(data[index]), index)
                sys.stdout.flush()
                ret[-1].append(data[index])
                index += 1

                if index == len(data) or ord(data[index]) == 255:
                    break
    return ret


def main():
    args = parse_args()
    host, port = args.ip_addr, args.port

    caffe.set_device(0)
    caffe.set_mode_gpu()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(2)
    connection = None

    loading_board = False
    pNet = caffe.Net(
        '../nets/pNet.prototxt', '../nets/pNet.300000.model', caffe.TEST)
    qNet = caffe.Net(
        '../nets/qNet.prototxt', '../nets/qNet.50000.model', caffe.TEST)
    brain = simpleSearchPlayer(
        pNet=pNet, qNet=qNet, pathCount=1024, depth=15)

    try:
        while True:
            connection, client_address = sock.accept()
            print('HAVE connection', client_address)
            while True:
                data = connection.recv(20).strip()

                # if connection was closed
                if not len(data):
                    break

                print("-------------------------------------")
                data = get_response(data)
                print("-------------------------------------")
                print("len - " + str(len(data)))

                for command in data:
                    print("|" + command + "|")
                    for c in command:
                        print("\"" + str(c) + "\"")
                    x, y = command[0], command[1]

                    # initialize board
                    if x == ord('i') and y == ord('n'):
                        board = np.ones((2, 19, 19), dtype=np.float32)

                    # board (load preloaded board state)
                    elif x == ord('b') and y == ord('o'):
                        loading_board = True
                        board = np.zeros((2, 19, 19))

                    # done (loading board state)
                    elif x == ord('d') and y == ord('o'):
                        loading_board = False

                    # move
                    elif x == ord('m') and y == ord('o'):
                        print("making move")
                        scores = brain.scoreMoves(board) ** 12
                        scores /= scores.sum()
                        pos = np.random.choice(scores.size, p=scores)
                        x, y = pos % 19, pos / 19
                        board[0, y, x] = -1
                        printBoard(board)
                        board[0, y, x] = 0
                        output = bytearray([x, y])
                        print("sending " + str(x) + " " + str(y))
                        connection.sendall(output)
                        print("move sent")

                    # make enemy move and my move
                    else:
                        if loading_board:
                            player = data[1]
                            print(player, data[1])
                            board[player, y, x] = 0
                            printBoard(board)
                        else:
                            if len(command) > 2:
                                player = command[2]
                            else:
                                player = 1
                            print('player', player)
                            board[player, y, x] = 0
                            printBoard(board)

            # disconnect(sock, connection)

    except KeyboardInterrupt:
        disconnect(sock, connection)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        disconnect(sock, connection)
        raise


if __name__ == "__main__":
    main()
